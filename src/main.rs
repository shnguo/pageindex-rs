use anyhow::{ Context, Result };
use async_recursion::async_recursion;
use base64::{ engine::general_purpose, Engine as _ };
use clap::Parser;
use lopdf::Document;
use reqwest::Client;
use serde::{ Deserialize, Serialize };
use std::path::{ Path, PathBuf };

mod db;
use db::LibraryIndex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    pdf_path: PathBuf,

    #[arg(long, default_value_t = 3)]
    max_depth: usize,

    #[arg(long, default_value_t = 5)]
    min_pages: usize,

    #[arg(long, default_value = "sqlite:pageindex.db?mode=rwc")]
    db_url: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DocumentNode {
    title: String,
    node_id: String,
    start_index: usize,
    end_index: usize,
    summary: String,
    #[serde(default)]
    has_children: Option<bool>,
    #[serde(default)]
    nodes: Vec<DocumentNode>,
}

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    InlineData {
        inline_data: InlineData,
    },
}

#[derive(Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Deserialize, Debug)]
struct CandidateContent {
    parts: Option<Vec<CandidatePart>>,
}

#[derive(Deserialize, Debug)]
struct CandidatePart {
    text: Option<String>,
}

/// Helper function to create a new PDF buffer containing only the specified page range.
/// Note: start_page and end_page are 1-indexed.
fn extract_pdf_pages(input_path: &Path, start_page: usize, end_page: usize) -> Result<Vec<u8>> {
    let doc = Document::load(input_path).context("Failed to load PDF for extraction")?;

    // lopdf uses 1-based indexing for page numbers
    let pages = doc.get_pages();
    let mut pages_to_keep = vec![];

    for (page_num, object_id) in pages.into_iter() {
        if (page_num as usize) >= start_page && (page_num as usize) <= end_page {
            pages_to_keep.push(object_id);
        }
    }

    if pages_to_keep.is_empty() {
        anyhow::bail!("No pages found in the specified range ({} - {})", start_page, end_page);
    }

    // Alternative approach: delete unwanted pages from the loaded document
    let mut doc_to_modify = Document::load(input_path)?;
    let all_pages = doc_to_modify.get_pages();
    let mut pages_to_delete = vec![];

    for (page_num, _) in all_pages {
        if (page_num as usize) < start_page || (page_num as usize) > end_page {
            pages_to_delete.push(page_num);
        }
    }

    doc_to_modify.delete_pages(&pages_to_delete);

    let mut buffer = Vec::new();
    doc_to_modify.save_to(&mut buffer).context("Failed to save extracted PDF to buffer")?;
    Ok(buffer)
}

/// Extract text for the specified page range.
fn extract_pdf_text(doc: &Document, start_page: usize, end_page: usize) -> Option<String> {
    let mut text = String::new();
    let pages = doc.get_pages();
    for page_num in start_page..=end_page {
        if let Some(_) = pages.get(&(page_num as u32)) {
            if let Ok(page_text) = doc.extract_text(&[page_num as u32]) {
                text.push_str(&page_text);
                text.push('\n');
            }
        }
    }
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

#[async_recursion]
async fn insert_nodes_recursively(
    db: &LibraryIndex,
    db_doc_id: &str,
    parent_id: Option<&str>,
    nodes: &[DocumentNode],
    pdf_doc: &Document
) -> Result<(), anyhow::Error> {
    for node in nodes {
        let unique_node_id = format!("{}_{}", db_doc_id, node.node_id);

        let child_ids: Vec<String> = node.nodes
            .iter()
            .map(|c| format!("{}_{}", db_doc_id, c.node_id))
            .collect();

        let content = extract_pdf_text(pdf_doc, node.start_index, node.end_index);

        db
            .insert_node(
                &unique_node_id,
                db_doc_id,
                parent_id,
                &node.title,
                &node.summary,
                content.as_deref(),
                node.start_index as i32,
                node.end_index as i32,
                node.has_children.unwrap_or(false),
                &child_ids
            ).await
            .context("Failed to insert node")?;

        if !node.nodes.is_empty() {
            insert_nodes_recursively(
                db,
                db_doc_id,
                Some(&unique_node_id),
                &node.nodes,
                pdf_doc
            ).await?;
        }
    }
    Ok(())
}

#[async_recursion]
async fn process_pdf_chunk(
    client: &Client,
    api_key: &str,
    pdf_path: &Path,
    abs_start_page: usize,
    abs_end_page: usize,
    current_depth: usize,
    max_depth: usize,
    min_pages: usize
) -> Result<Vec<DocumentNode>> {
    let num_pages = (abs_end_page + 1).saturating_sub(abs_start_page);

    println!(
        "{:indent$}Processing chunk: pages {} to {} (Depth: {}, {} pages)",
        "",
        abs_start_page,
        abs_end_page,
        current_depth,
        num_pages,
        indent = current_depth * 2
    );

    // Hard Limit check
    if current_depth >= max_depth {
        println!(
            "{:indent$}Reached max depth ({}), stopping recursion.",
            "",
            max_depth,
            indent = current_depth * 2
        );
        return Ok(vec![]);
    }
    if num_pages < min_pages && current_depth > 0 {
        println!(
            "{:indent$}Chunk too small ({} pages < min {}), stopping recursion.",
            "",
            num_pages,
            min_pages,
            indent = current_depth * 2
        );
        return Ok(vec![]);
    }

    let pdf_bytes = extract_pdf_pages(pdf_path, abs_start_page, abs_end_page)?;
    let b64_pdf = general_purpose::STANDARD.encode(&pdf_bytes);

    let url =
        format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={}", api_key);

    let prompt = format!(
        "Extract the document structure of this PDF chunk. Output a JSON forest structure, with the top level being a list of nodes. \
        Each node contains: 'title', 'node_id', 'start_index' (number, RELATIVE starting page in this chunk, where chunk start = 1), \
        'end_index' (relative ending page), 'summary', and 'has_children'. \
        Set 'has_children' to true only if this section clearly contains distinct subdivisions. \
        Do not include sub-nodes; just return the top-level subdivisions for this text chunk."
    );

    let schema: serde_json::Value =
        serde_json::json!({
        "type": "ARRAY",
        "description": "A list of document nodes representing the top-level structure of the PDF chunk.",
        "items": {
            "type": "OBJECT",
            "properties": {
                "title": { "type": "STRING", "description": "The title or heading of the section." },
                "node_id": { "type": "STRING", "description": "A unique identifier for this node." },
                "start_index": { "type": "INTEGER", "description": "The relative starting page number within this chunk (1-indexed)." },
                "end_index": { "type": "INTEGER", "description": "The relative ending page number within this chunk." },
                "summary": { "type": "STRING", "description": "A brief summary of the section's contents." },
                "has_children": { "type": "BOOLEAN", "description": "True if this section contains further major subdivisions." }
            },
            "required": ["title", "node_id", "start_index", "end_index", "summary", "has_children"]
        }
    });

    let request_body = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![
                Part::InlineData {
                    inline_data: InlineData {
                        mime_type: "application/pdf".to_string(),
                        data: b64_pdf,
                    },
                },
                Part::Text {
                    text: prompt,
                }
            ],
        }],
        system_instruction: None,
        generation_config: Some(GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema),
        }),
    };

    let res = client
        .post(&url)
        .json(&request_body)
        .send().await
        .context("Failed to send request to Gemini API")?;

    if !res.status().is_success() {
        let error_text = res.text().await?;
        anyhow::bail!("API request failed: {}", error_text);
    }

    let response: GenerateContentResponse = res.json().await.context("Failed to parse response")?;

    let text_content = response.candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .and_then(|p| p.into_iter().next())
        .and_then(|p| p.text)
        .unwrap_or_else(|| "[]".to_string());

    let text_content = text_content.trim();
    let text_content = if text_content.starts_with("```json") {
        text_content
            .strip_prefix("```json")
            .unwrap_or(text_content)
            .strip_suffix("```")
            .unwrap_or(text_content)
            .trim()
    } else {
        text_content
    };

    let mut nodes: Vec<DocumentNode> = serde_json
        ::from_str(text_content)
        .unwrap_or_else(|_| vec![]);

    // Convert relative coordinates to absolute and process children
    for node in nodes.iter_mut() {
        // Convert relative to absolute
        // Note: if LLM returns 1, and chunk starts at 5, actual is 5 + 1 - 1 = 5.
        let actual_start = abs_start_page + node.start_index.saturating_sub(1);
        let actual_end = abs_start_page + node.end_index.saturating_sub(1);

        // Ensure bounds are sane
        let actual_start = actual_start.max(abs_start_page).min(abs_end_page);
        let actual_end = actual_end.max(actual_start).min(abs_end_page);

        node.start_index = actual_start;
        node.end_index = actual_end;

        if node.has_children.unwrap_or(false) {
            let child_nodes = process_pdf_chunk(
                client,
                api_key,
                pdf_path,
                node.start_index,
                node.end_index,
                current_depth + 1,
                max_depth,
                min_pages
            ).await?;
            node.nodes = child_nodes;
        }
    }

    Ok(nodes)
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY not found in .env")?;

    let args = Args::parse();

    let db = LibraryIndex::new(&args.db_url).await.context("Failed to connect to database")?;
    db.init_tables().await.context("Failed to initialize database tables")?;

    if !args.pdf_path.exists() {
        anyhow::bail!("PDF file not found: {:?}", args.pdf_path);
    }

    let doc = Document::load(&args.pdf_path).context(
        "Failed to load PDF to determine total pages"
    )?;
    let total_pages = doc.get_pages().len();

    println!("Starting recursive extraction on {:?} ({} pages)", args.pdf_path, total_pages);
    println!("Limits: Max Depth={}, Min Pages={}", args.max_depth, args.min_pages);

    let client = Client::new();

    let root_nodes = process_pdf_chunk(
        &client,
        &api_key,
        &args.pdf_path,
        1,
        total_pages,
        0,
        args.max_depth,
        args.min_pages
    ).await?;

    println!("\n==================================");
    println!("Extraction Complete!");
    println!("==================================\n");
    let pretty_json = serde_json::to_string_pretty(&root_nodes)?;
    println!("{}", pretty_json);

    println!("\nSaving to database...");
    let db_doc_id = uuid::Uuid::new_v4().to_string();
    let doc_title = args.pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown Document");

    // Generate an overall summary if root_nodes has anything, or just a default text
    let overall_summary = if !root_nodes.is_empty() {
        Some(
            root_nodes
                .iter()
                .map(|n| n.summary.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        )
    } else {
        None
    };

    db.insert_document(&db_doc_id, doc_title, overall_summary.as_deref()).await?;
    insert_nodes_recursively(&db, &db_doc_id, None, &root_nodes, &doc).await?;
    println!("Saved to database successfully.");

    Ok(())
}
