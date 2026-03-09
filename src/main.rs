use anyhow::{ Context, Result };
use async_recursion::async_recursion;
use base64::{ Engine as _, engine::general_purpose };
use clap::{ Parser, Subcommand };
use lopdf::Document;
use reqwest::Client;
use serde::{ Deserialize, Serialize };
use sha2::{ Digest, Sha256 };
use std::fs::{ self, File };
use std::io::Read;
use std::path::{ Path, PathBuf };
use tracing::{info, warn};
mod db;
use db::{ HashCheckResult, LibraryIndex };
use sqlx::{ Sqlite, Transaction };

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value = "sqlite:pageindex.db?mode=rwc")]
    db_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Parse and index a new PDF document into the database
    Index {
        #[arg(short, long)]
        pdf_path: PathBuf,

        #[arg(long, default_value_t = 3)]
        max_depth: usize,

        #[arg(long, default_value_t = 5)]
        min_pages: usize,
    },
    /// Search documents by keyword in their summary
    Search {
        keyword: String,
    },
    /// Get top level nodes for a document
    TopNodes {
        document_id: String,
    },
    /// Get details of specific nodes and their children
    Nodes {
        node_ids: Vec<String>,
    },
    /// Read raw content of a specific leaf node
    ReadContent {
        node_id: String,
    },
    /// Resolve a cross-reference within a document
    ResolveRef {
        reference_text: String,
        document_id: String,
    },
    /// List images and tables associated with a node
    ListAssets {
        node_id: String,
    },
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
    has_images: Option<bool>,
    #[serde(default)]
    has_tables: Option<bool>,
    #[serde(default)]
    references: Vec<String>,
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

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f32,
    response_format: Option<OpenAiResponseFormat>,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: Vec<OpenAiContentPart>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum OpenAiContentPart {
    #[serde(rename = "text")] Text {
        text: String,
    },
    #[serde(rename = "image_url")] ImageUrl {
        image_url: OpenAiImageUrl,
    },
}

#[derive(Serialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Serialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize, Debug)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

use image::ImageFormat;
use pdfium_render::prelude::*;

/// Helper function to create a new PDF buffer containing only the specified page range.
/// Note: start_page and end_page are 1-indexed.
fn extract_pdf_pages(input_path: &Path, start_page: usize, end_page: usize) -> Result<Vec<u8>> {
    let mut doc = Document::load(input_path).context("Failed to load PDF for extraction")?;

    // Collect pages to delete (outside the desired range)
    let all_pages = doc.get_pages();
    let mut pages_to_delete = vec![];
    let mut has_pages_in_range = false;

    for (page_num, _) in all_pages {
        if (page_num as usize) < start_page || (page_num as usize) > end_page {
            pages_to_delete.push(page_num);
        } else {
            has_pages_in_range = true;
        }
    }

    if !has_pages_in_range {
        anyhow::bail!("No pages found in the specified range ({} - {})", start_page, end_page);
    }

    doc.delete_pages(&pages_to_delete);

    let mut buffer = Vec::new();
    doc.save_to(&mut buffer).context("Failed to save extracted PDF to buffer")?;
    Ok(buffer)
}

/// Render a single PDF page to JPEG. Loads Pdfium + document each call.
/// For batch rendering, prefer render_pdf_pages_batch.
fn render_pdf_page_to_jpeg(input_path: &Path, page_number: usize) -> Result<Vec<u8>> {
    let pdfium = Pdfium::default();
    let document = pdfium.load_pdf_from_file(input_path, None)?;

    let page = document.pages().get((page_number - 1) as PdfPageIndex)?;
    let bitmap = page.render_with_config(&PdfRenderConfig::new().set_target_width(2000))?;
    let image = bitmap.as_image();

    let mut buffer = std::io::Cursor::new(Vec::new());
    image.write_to(&mut buffer, ImageFormat::Jpeg)?;
    Ok(buffer.into_inner())
}

/// Batch-render multiple PDF pages to JPEG, loading Pdfium and the document only once.
/// Returns a Vec of (page_number, jpeg_bytes) for each successfully rendered page.
fn render_pdf_pages_batch(input_path: &Path, pages: std::ops::RangeInclusive<usize>) -> Vec<(usize, Vec<u8>)> {
    let pdfium = Pdfium::default();
    let document = match pdfium.load_pdf_from_file(input_path, None) {
        Ok(doc) => doc,
        Err(e) => {
            warn!("[Batch Render] Failed to load PDF for batch rendering: {}", e);
            return vec![];
        }
    };

    let mut results = Vec::new();
    for page_num in pages {
        match document.pages().get((page_num - 1) as PdfPageIndex) {
            Ok(page) => {
                match page.render_with_config(&PdfRenderConfig::new().set_target_width(2000)) {
                    Ok(bitmap) => {
                        let image = bitmap.as_image();
                        let mut buffer = std::io::Cursor::new(Vec::new());
                        if image.write_to(&mut buffer, ImageFormat::Jpeg).is_ok() {
                            results.push((page_num, buffer.into_inner()));
                        }
                    }
                    Err(e) => warn!("[Batch Render] Failed to render page {}: {}", page_num, e),
                }
            }
            Err(e) => warn!("[Batch Render] Failed to get page {}: {}", page_num, e),
        }
    }
    results
}

/// Retry an async HTTP request with exponential backoff.
/// Retries on transient errors (network, 429, 5xx). Fails immediately on 4xx (except 429).
async fn send_with_retry(
    _client: &Client,
    build_request: impl Fn() -> reqwest::RequestBuilder,
    context_msg: &str,
) -> Result<reqwest::Response> {
    let max_retries = 3u32;
    let mut last_err = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            let delay = std::time::Duration::from_secs(1 << (attempt - 1)); // 1s, 2s, 4s
            warn!("{}: attempt {}/{} after {:?} delay", context_msg, attempt + 1, max_retries + 1, delay);
            tokio::time::sleep(delay).await;
        }

        let res = match build_request().send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(format!("Network error: {}", e));
                continue;
            }
        };

        let status = res.status();
        if status.is_success() {
            return Ok(res);
        }

        // Retry on 429 (rate limit) and 5xx (server errors)
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            last_err = Some(format!("HTTP {}", status));
            continue;
        }

        // Non-retryable client error (400, 401, 403, etc.)
        let error_text = res.text().await.unwrap_or_default();
        anyhow::bail!("{}: HTTP {} - {}", context_msg, status, error_text);
    }

    anyhow::bail!("{}: all {} retries exhausted. Last error: {}",
        context_msg, max_retries + 1, last_err.unwrap_or_default())
}

/// Extract text for the specified page range using local PaddleOCR-VL via MLX.
// The MLX PaddleOCR-VL server requires images to process, not PDF binaries.
// We render each requested page of the PDF into a JPEG and attach it to the request.
async fn extract_pdf_text_via_ocr(
    client: &Client,
    input_path: &Path,
    start_page: usize,
    end_page: usize
) -> Result<Option<String>> {
    let url = "http://127.0.0.1:8080/chat/completions";
    let mut full_text = String::new();
    let prompt =
        "Extract all the text from this document image accurately. Do not output anything else other than the text.";

    for page_num in start_page..=end_page {
        info!("    [OCR] Rendering page {} to JPEG...", page_num);
        // Render the page to a JPEG buffer
        let jpeg_bytes = render_pdf_page_to_jpeg(input_path, page_num).context(
            format!("Failed to render page {} to JPEG", page_num)
        )?;

        let b64_jpeg = general_purpose::STANDARD.encode(&jpeg_bytes);

        // One page per request for reliability and following MLX server image limits
        let request_body = OpenAiChatRequest {
            model: "mlx-community/PaddleOCR-VL-1.5-4bit".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: vec![
                    OpenAiContentPart::Text {
                        text: prompt.to_string(),
                    },
                    OpenAiContentPart::ImageUrl {
                        image_url: OpenAiImageUrl {
                            url: format!("data:image/jpeg;base64,{}", b64_jpeg),
                        },
                    }
                ],
            }],
            temperature: 0.0,
            response_format: None,
        };

        info!("    [OCR] Sending page {} to local MLX server...", page_num);
        let request_body_json = serde_json::to_value(&request_body)
            .context("Failed to serialize OCR request")?;
        let ocr_url = url.to_string();
        let res = send_with_retry(
            client,
            || client.post(&ocr_url).json(&request_body_json),
            &format!("OCR page {}", page_num),
        ).await?;

        let response: OpenAiChatResponse = res
            .json().await
            .context("Failed to parse OCR response")?;
        if let Some(choice) = response.choices.first() {
            if let Some(ref text) = choice.message.content {
                info!("    [OCR] Successfully extracted text for page {}.", page_num);
                full_text.push_str(text);
                full_text.push_str("\n\n");
            }
        }
    }

    Ok(if full_text.trim().is_empty() { None } else { Some(full_text) })
}

/// Extract table content from a pre-rendered JPEG page image via the local OCR service.
async fn extract_table_text_via_ocr(
    client: &Client,
    jpeg_bytes: &[u8]
) -> Result<Option<String>> {
    let url = "http://127.0.0.1:8080/chat/completions";
    let b64_jpeg = general_purpose::STANDARD.encode(jpeg_bytes);
    let prompt = "Extract all tables from this image as markdown tables. If there are no tables, respond with exactly 'NONE'.";

    let request_body = OpenAiChatRequest {
        model: "mlx-community/PaddleOCR-VL-1.5-4bit".to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: vec![
                OpenAiContentPart::Text { text: prompt.to_string() },
                OpenAiContentPart::ImageUrl {
                    image_url: OpenAiImageUrl {
                        url: format!("data:image/jpeg;base64,{}", b64_jpeg),
                    },
                }
            ],
        }],
        temperature: 0.0,
        response_format: None,
    };

    let res = client.post(url).json(&request_body).send().await
        .context("Failed to send table extraction request")?;

    if !res.status().is_success() {
        return Ok(None);
    }

    let response: OpenAiChatResponse = res.json().await
        .context("Failed to parse table extraction response")?;

    if let Some(choice) = response.choices.first() {
        if let Some(ref text) = choice.message.content {
            if text.trim() != "NONE" {
                return Ok(Some(text.clone()));
            }
        }
    }
    Ok(None)
}

/// Extract image/figure descriptions from a pre-rendered JPEG page image via the local OCR service.
async fn extract_image_description_via_ocr(
    client: &Client,
    jpeg_bytes: &[u8]
) -> Result<Option<String>> {
    let url = "http://127.0.0.1:8080/chat/completions";
    let b64_jpeg = general_purpose::STANDARD.encode(jpeg_bytes);
    let prompt = "Describe each image, figure, chart, or diagram on this page. Include what the visual shows and any labels or captions. If there are no images or figures, respond with exactly 'NONE'.";

    let request_body = OpenAiChatRequest {
        model: "mlx-community/PaddleOCR-VL-1.5-4bit".to_string(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: vec![
                OpenAiContentPart::Text { text: prompt.to_string() },
                OpenAiContentPart::ImageUrl {
                    image_url: OpenAiImageUrl {
                        url: format!("data:image/jpeg;base64,{}", b64_jpeg),
                    },
                }
            ],
        }],
        temperature: 0.0,
        response_format: None,
    };

    let res = client.post(url).json(&request_body).send().await
        .context("Failed to send image description request")?;

    if !res.status().is_success() {
        return Ok(None);
    }

    let response: OpenAiChatResponse = res.json().await
        .context("Failed to parse image description response")?;

    if let Some(choice) = response.choices.first() {
        if let Some(ref text) = choice.message.content {
            if text.trim() != "NONE" {
                return Ok(Some(text.clone()));
            }
        }
    }
    Ok(None)
}

fn generate_node_id(db_doc_id: &str, parent_id: &str, raw_node_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}_{}_{}", db_doc_id, parent_id, raw_node_id));
    format!("{:x}", hasher.finalize())
}

#[async_recursion]
async fn insert_nodes_recursively(
    client: &Client,
    tx: &mut Transaction<'static, Sqlite>,
    db_doc_id: &str,
    parent_id: Option<&str>,
    nodes: &[DocumentNode],
    pdf_path: &Path,
    assets_dir: &Path
) -> Result<(), anyhow::Error> {
    for node in nodes {
        let unique_node_id = generate_node_id(
            db_doc_id,
            parent_id.unwrap_or("root"),
            &node.node_id
        );

        let child_ids: Vec<String> = node.nodes
            .iter()
            .map(|c| generate_node_id(db_doc_id, &unique_node_id, &c.node_id))
            .collect();

        // Only run OCR extraction if this is a leaf node (no children)
        let is_leaf = node.nodes.is_empty();
        let content = if is_leaf {
            extract_pdf_text_via_ocr(client, pdf_path, node.start_index, node.end_index).await?
        } else {
            None
        };

        let has_images = node.has_images.unwrap_or(false);
        let has_tables = node.has_tables.unwrap_or(false);

        LibraryIndex::insert_node_tx(
            tx,
            &unique_node_id,
            db_doc_id,
            parent_id,
            &node.title,
            &node.summary,
            content.as_deref(),
            node.start_index as i32,
            node.end_index as i32,
            !is_leaf,
            has_images,
            has_tables,
            &child_ids
        ).await
            .context("Failed to insert node")?;

        // Insert cross-references
        for ref_text in &node.references {
            LibraryIndex::insert_reference_tx(tx, &unique_node_id, ref_text, db_doc_id).await
                .context("Failed to insert reference")?;
        }

        // Extract images/tables for leaf nodes (batch-render for efficiency)
        if is_leaf && (has_images || has_tables) {
            info!("  [Assets] Extracting assets for node '{}' (pages {}-{})", node.title, node.start_index, node.end_index);
            let rendered_pages = render_pdf_pages_batch(pdf_path, node.start_index..=node.end_index);
            for (page_num, jpeg_bytes) in &rendered_pages {
                let asset_filename = format!("{}_{}.jpg", unique_node_id, page_num);
                let asset_path = assets_dir.join(&asset_filename);
                if let Err(e) = fs::write(&asset_path, jpeg_bytes) {
                    info!("  [Assets] Warning: Failed to write asset file: {}", e);
                    continue;
                }
                let asset_path_str = asset_path.to_string_lossy().to_string();

                // Run table + image OCR concurrently when both are needed
                let (table_result, image_result) = tokio::join!(
                    async {
                        if has_tables {
                            extract_table_text_via_ocr(client, jpeg_bytes).await.unwrap_or(None)
                        } else {
                            None
                        }
                    },
                    async {
                        if has_images {
                            extract_image_description_via_ocr(client, jpeg_bytes).await.unwrap_or(None)
                        } else {
                            None
                        }
                    }
                );

                if let Some(ref text) = table_result {
                    LibraryIndex::insert_asset_tx(
                        tx, &unique_node_id, "table",
                        Some(&format!("Table from page {}", page_num)),
                        Some(*page_num as i32),
                        Some(&asset_path_str),
                        Some(text)
                    ).await.context("Failed to insert table asset")?;
                }

                if has_images {
                    LibraryIndex::insert_asset_tx(
                        tx, &unique_node_id, "image",
                        image_result.as_deref(),
                        Some(*page_num as i32),
                        Some(&asset_path_str),
                        None
                    ).await.context("Failed to insert image asset")?;
                }
            }
        }

        if !is_leaf {
            insert_nodes_recursively(
                client,
                tx,
                db_doc_id,
                Some(&unique_node_id),
                &node.nodes,
                pdf_path,
                assets_dir
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

    info!(
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
        info!(
            "{:indent$}Reached max depth ({}), stopping recursion.",
            "",
            max_depth,
            indent = current_depth * 2
        );
        return Ok(vec![]);
    }
    if num_pages < min_pages && current_depth > 0 {
        info!(
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
        'end_index' (relative ending page), 'summary', 'has_children', 'has_images', 'has_tables', and 'references'. \
        Set 'has_children' to true only if this section clearly contains distinct subdivisions. \
        Set 'has_images' to true if this section contains images, figures, or diagrams. \
        Set 'has_tables' to true if this section contains data tables. \
        In 'references', list any cross-references found (e.g. 'Appendix G', 'Table 5.3', 'Section 3.1'). \
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
                "has_children": { "type": "BOOLEAN", "description": "True if this section contains further major subdivisions." },
                "has_images": { "type": "BOOLEAN", "description": "True if this section contains images, figures, or diagrams." },
                "has_tables": { "type": "BOOLEAN", "description": "True if this section contains data tables." },
                "references": { "type": "ARRAY", "items": { "type": "STRING" }, "description": "Cross-references found in this section (e.g. 'Appendix G', 'Table 5.3')." }
            },
            "required": ["title", "node_id", "start_index", "end_index", "summary", "has_children", "has_images", "has_tables", "references"]
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
                Part::Text { text: prompt }
            ],
        }],
        system_instruction: None,
        generation_config: Some(GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema),
        }),
    };

    let request_body_json = serde_json::to_value(&request_body)
        .context("Failed to serialize Gemini request")?;
    let gemini_url = url.clone();
    let res = send_with_retry(
        client,
        || client.post(&gemini_url).json(&request_body_json),
        &format!("Gemini API (pages {}-{})", abs_start_page, abs_end_page),
    ).await?;

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

    let mut nodes: Vec<DocumentNode> = match serde_json::from_str(text_content) {
        Ok(parsed) => parsed,
        Err(e) => {
            warn!(
                "Failed to parse Gemini JSON response for pages {}-{}: {}\nRaw response: {}",
                abs_start_page, abs_end_page, e, text_content
            );
            vec![]
        }
    };

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

async fn run_index(
    db: &LibraryIndex,
    api_key: &str,
    pdf_path: PathBuf,
    max_depth: usize,
    min_pages: usize
) -> Result<()> {
    if !pdf_path.exists() {
        anyhow::bail!("PDF file not found: {:?}", pdf_path);
    }

    let absolute_path_str = match std::fs::canonicalize(&pdf_path) {
        Ok(path) => path.to_string_lossy().to_string(),
        Err(_) => pdf_path.to_string_lossy().to_string(),
    };

    info!("Calculating SHA-256 hash for document...");
    let mut file = File::open(&pdf_path).context("Failed to open PDF for hashing")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let file_hash = format!("{:x}", hasher.finalize());
    info!("Document Hash: {}", file_hash);

    match db.check_document_hash(&absolute_path_str, &file_hash).await? {
        HashCheckResult::Match => {
            info!("✅ Document is unchanged. Skipping extraction.");
            return Ok(());
        }
        HashCheckResult::Mismatch(old_doc_id) => {
            info!("🔄 Document has been modified. Pruning old data (ID: {})...", old_doc_id);
            db.prune_document(&old_doc_id).await.context("Failed to prune old document")?;
        }
        HashCheckResult::NotFound => {
            info!("📄 New document detected. Proceeding with extraction.");
        }
    }

    let doc = Document::load(&pdf_path).context("Failed to load PDF to determine total pages")?;
    let total_pages = doc.get_pages().len();

    info!("Starting recursive extraction on {:?} ({} pages)", pdf_path, total_pages);
    info!("Limits: Max Depth={}, Min Pages={}", max_depth, min_pages);

    let client = Client::new();

    // OCR health check: verify the local OCR server is reachable before starting
    info!("Checking OCR server availability...");
    match client.get("http://127.0.0.1:8080/").send().await {
        Ok(_) => info!("✅ OCR server is reachable."),
        Err(_) => {
            anyhow::bail!(
                "❌ OCR server is not running at http://127.0.0.1:8080.\n\
                 Please start it first:\n  cd ocr-service && uv run python main.py"
            );
        }
    }

    let root_nodes = process_pdf_chunk(
        &client,
        api_key,
        &pdf_path,
        1,
        total_pages,
        0,
        max_depth,
        min_pages
    ).await?;

    info!("\n==================================");
    info!("Extraction Complete!");
    info!("==================================\n");
    let pretty_json = serde_json::to_string_pretty(&root_nodes)?;
    println!("{}", pretty_json);

    info!("\nSaving to database (inside transaction)...");
    let db_doc_id = uuid::Uuid::new_v4().to_string();
    let doc_title = pdf_path
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

    // Insert document record (outside transaction — standalone first write)
    db.insert_document(
        &db_doc_id,
        doc_title,
        overall_summary.as_deref(),
        Some(&absolute_path_str),
        Some(&file_hash)
    ).await.context("Failed to insert document")?;

    // Create assets directory next to the database
    let assets_dir = PathBuf::from("assets");
    fs::create_dir_all(&assets_dir).context("Failed to create assets directory")?;

    // Wrap all node/reference/asset inserts in a proper sqlx transaction
    let mut tx = db.begin().await.context("Failed to begin transaction")?;
    insert_nodes_recursively(&client, &mut tx, &db_doc_id, None, &root_nodes, &pdf_path, &assets_dir).await
        .context("Failed to insert nodes (transaction rolled back)")?;
    tx.commit().await.context("Failed to commit transaction")?;
    info!("Saved to database successfully.");

    // Link cross-references after commit (reads the committed data)
    let linked = db.link_references(&db_doc_id).await.unwrap_or(0);
    if linked > 0 {
        info!("Linked {} cross-references.", linked);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber
        ::fmt()
        .with_writer(std::io::stdout)
        .with_env_filter(
            tracing_subscriber::EnvFilter
                ::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let db = LibraryIndex::new(&cli.db_url).await.context("Failed to connect to database")?;
    db.init_tables().await.context("Failed to initialize database tables")?;

    match cli.command {
        Commands::Index { pdf_path, max_depth, min_pages } => {
            let api_key = std::env
                ::var("GEMINI_API_KEY")
                .context("GEMINI_API_KEY not found in .env")?;
            run_index(&db, &api_key, pdf_path, max_depth, min_pages).await?;
        }
        Commands::Search { keyword } => {
            println!("Searching for keyword: {}", keyword);

            // Tier 1: FTS5 on documents (with prefix matching)
            let docs = db.search_documents_by_summary(&keyword).await?;
            if !docs.is_empty() {
                println!("[Document matches] Found {} documents:", docs.len());
                for doc in &docs {
                    println!("----------------------------------------");
                    println!("ID: {}", doc.id);
                    println!("Title: {}", doc.title);
                    if let Some(ref file_path) = doc.file_path {
                        println!("File Path: {}", file_path);
                    }
                    if let Some(ref summary) = doc.overall_summary {
                        println!("Summary: {}", summary);
                    }
                }
                println!("----------------------------------------");
            }

            // Tier 2: FTS5 on document_nodes (with prefix matching)
            let nodes = db.search_nodes(&keyword).await?;
            if !nodes.is_empty() {
                println!("[Section matches] Found {} matching sections:", nodes.len());
                for node in &nodes {
                    println!("----------------------------------------");
                    println!("Node ID: {}", node.node_id);
                    println!("Title: {}", node.title);
                    println!("Summary: {}", node.summary);
                    println!("Has Children: {}", node.has_children);
                }
                println!("----------------------------------------");
            }

            // Tier 3: LIKE fallback if both FTS tiers returned nothing
            if docs.is_empty() && nodes.is_empty() {
                println!("[FTS returned no results, trying fuzzy LIKE fallback...]");
                let fuzzy_docs = db.search_documents_fuzzy(&keyword).await?;
                let fuzzy_nodes = db.search_nodes_fuzzy(&keyword).await?;

                if fuzzy_docs.is_empty() && fuzzy_nodes.is_empty() {
                    println!("No documents or sections found matching the keyword.");
                } else {
                    if !fuzzy_docs.is_empty() {
                        println!("[Fuzzy document matches] Found {} documents:", fuzzy_docs.len());
                        for doc in &fuzzy_docs {
                            println!("----------------------------------------");
                            println!("ID: {}", doc.id);
                            println!("Title: {}", doc.title);
                            if let Some(ref summary) = doc.overall_summary {
                                println!("Summary: {}", summary);
                            }
                        }
                        println!("----------------------------------------");
                    }
                    if !fuzzy_nodes.is_empty() {
                        println!("[Fuzzy section matches] Found {} sections:", fuzzy_nodes.len());
                        for node in &fuzzy_nodes {
                            println!("----------------------------------------");
                            println!("Node ID: {}", node.node_id);
                            println!("Title: {}", node.title);
                            println!("Summary: {}", node.summary);
                        }
                        println!("----------------------------------------");
                    }
                }
            }
        }
        Commands::TopNodes { document_id } => {
            println!("Retrieving top-level nodes for document ID: {}", document_id);
            let nodes = db.get_top_level_nodes(&[document_id]).await?;

            if nodes.is_empty() {
                println!("No top-level nodes found for this document.");
            } else {
                println!("Found {} top-level nodes:", nodes.len());
                for node in nodes {
                    println!("----------------------------------------");
                    println!("Node ID: {}", node.node_id);
                    println!("Title: {}", node.title);
                    println!("Summary: {}", node.summary);
                    println!("Has Children: {}", node.has_children);
                    if node.has_children {
                        println!("Child IDs: {:?}", node.child_ids.0);
                        if !node.child_ids.0.is_empty() {
                            let child_nodes = db.get_nodes_by_ids(&node.child_ids.0).await?;
                            println!("Child Nodes:");
                            for child in child_nodes {
                                println!("  - ID: {}", child.node_id);
                                println!("    Title: {}", child.title);
                                println!("    Summary: {}", child.summary);
                            }
                        }
                    }
                }
                println!("----------------------------------------");
            }
        }
        Commands::Nodes { node_ids } => {
            println!("Retrieving details for {} nodes...", node_ids.len());
            let nodes = db.get_full_nodes_by_ids(&node_ids).await?;

            if nodes.is_empty() {
                println!("No nodes found for the provided IDs.");
            } else {
                for node in nodes {
                    println!("========================================");
                    println!("Node ID: {}", node.node_id);
                    println!("Document ID: {}", node.document_id);
                    if let Some(parent) = node.parent_id {
                        println!("Parent ID: {}", parent);
                    }
                    println!("Title: {}", node.title);
                    if let (Some(start), Some(end)) = (node.start_index, node.end_index) {
                        println!("Pages: {} - {}", start, end);
                    }
                    println!("Summary: {}", node.summary);

                    if let Some(content) = node.content {
                        let short_content: String = content.chars().take(200).collect();
                        println!("\nContent preview: {}...", short_content.replace("\n", " "));
                    }

                    println!("\nHas Children: {}", node.has_children);
                    if node.has_children && !node.child_ids.0.is_empty() {
                        let child_nodes = db.get_nodes_by_ids(&node.child_ids.0).await?;
                        println!("\nChild Nodes ({}):", child_nodes.len());
                        for child in child_nodes {
                            println!("  - ID: {}", child.node_id);
                            println!("    Title: {}", child.title);
                            println!("    Summary: {}", child.summary);
                        }
                    }
                    println!("========================================");
                }
            }
        }
        Commands::ReadContent { node_id } => {
            let node_content = db.read_node_content(&node_id).await?;
            match node_content {
                Some(nc) => {
                    println!("Node: {} ({})", nc.title, nc.node_id);
                    match nc.content {
                        Some(text) => println!("{}", text),
                        None => println!("(This node has no raw text content. It may be a parent node — try drilling into its children.)"),
                    }
                }
                None => println!("Node not found or associated file is missing."),
            }
        }
        Commands::ResolveRef { reference_text, document_id } => {
            println!("Resolving reference '{}' in document {}...", reference_text, document_id);
            let nodes = db.resolve_reference(&reference_text, &document_id).await?;
            if nodes.is_empty() {
                println!("No matching sections found for this reference.");
            } else {
                println!("Found {} matching sections:", nodes.len());
                for node in nodes {
                    println!("----------------------------------------");
                    println!("Node ID: {}", node.node_id);
                    println!("Title: {}", node.title);
                    println!("Summary: {}", node.summary);
                    println!("Has Children: {}", node.has_children);
                }
                println!("----------------------------------------");
            }
        }
        Commands::ListAssets { node_id } => {
            println!("Assets for node {}:", node_id);
            let assets = db.get_assets_for_node(&node_id).await?;
            if assets.is_empty() {
                println!("No assets found for this node.");
            } else {
                println!("Found {} assets:", assets.len());
                for asset in assets {
                    println!("----------------------------------------");
                    println!("Type: {}", asset.asset_type);
                    if let Some(desc) = asset.description {
                        println!("Description: {}", desc);
                    }
                    if let Some(page) = asset.page_number {
                        println!("Page: {}", page);
                    }
                    if let Some(path) = asset.file_path {
                        println!("File: {}", path);
                    }
                    if let Some(table) = asset.table_text {
                        println!("Table Content:\n{}", table);
                    }
                }
                println!("----------------------------------------");
            }
        }
    }

    Ok(())
}
