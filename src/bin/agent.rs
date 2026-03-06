use anyhow::{ Context, Result };
use reqwest::Client;
use serde::{ Deserialize, Serialize };
use serde_json::Value;
use std::io::{ self, Write };
use std::process::Command;
use dotenvy::dotenv;

// ---------------------------------------------------------------------------
// Gemini API Schemas
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Content {
    pub role: String, // "user" or "model" or "function"
    pub parts: Vec<Part>,
}

use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionCall")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionResponse")]
    pub function_response: Option<FunctionResponse>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionCall {
    pub name: String,
    pub args: Value,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionResponse {
    pub name: String,
    pub response: Value,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Tool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
}

#[derive(Deserialize, Debug, Clone)]
struct GenerateContentResponse {
    pub candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize, Debug, Clone)]
struct Candidate {
    pub content: Option<Content>,
}

// ---------------------------------------------------------------------------
// Tool Execution Logic
// ---------------------------------------------------------------------------

fn execute_tool(name: &str, args: &Value) -> Value {
    println!("\n[Agent] executing tool: {} with args: {}", name, args);
    let mut cmd = Command::new("cargo");
    cmd.arg("run").arg("--bin").arg("pageindex-rs").arg("-q").arg("--"); // Explicitly run the main app bin

    match name {
        "search_library" => {
            if let Some(keyword) = args.get("keyword").and_then(|v| v.as_str()) {
                cmd.arg("search").arg(keyword);
            } else {
                return serde_json::json!({"error": "Missing 'keyword' argument"});
            }
        }
        "read_toc" => {
            if let Some(document_id) = args.get("document_id").and_then(|v| v.as_str()) {
                cmd.arg("top-nodes").arg(document_id);
            } else {
                return serde_json::json!({"error": "Missing 'document_id' argument"});
            }
        }
        "drill_down" => {
            if let Some(node_id) = args.get("node_id").and_then(|v| v.as_str()) {
                cmd.arg("nodes").arg(node_id);
            } else {
                return serde_json::json!({"error": "Missing 'node_id' argument"});
            }
        }
        "read_content" => {
            if let Some(node_id) = args.get("node_id").and_then(|v| v.as_str()) {
                cmd.arg("read-content").arg(node_id);
            } else {
                return serde_json::json!({"error": "Missing 'node_id' argument"});
            }
        }
        "resolve_reference" => {
            let ref_text = args.get("reference_text").and_then(|v| v.as_str());
            let doc_id = args.get("document_id").and_then(|v| v.as_str());
            if let (Some(ref_text), Some(doc_id)) = (ref_text, doc_id) {
                cmd.arg("resolve-ref").arg(ref_text).arg(doc_id);
            } else {
                return serde_json::json!({"error": "Missing 'reference_text' or 'document_id' argument"});
            }
        }
        "list_assets" => {
            if let Some(node_id) = args.get("node_id").and_then(|v| v.as_str()) {
                cmd.arg("list-assets").arg(node_id);
            } else {
                return serde_json::json!({"error": "Missing 'node_id' argument"});
            }
        }
        _ => {
            return serde_json::json!({"error": format!("Unknown tool: {}", name)});
        }
    }

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if output.status.success() {
                serde_json::json!({"result": stdout})
            } else {
                serde_json::json!({"error": stderr})
            }
        }
        Err(e) => serde_json::json!({"error": format!("Failed to execute command: {}", e)}),
    }
}

// ---------------------------------------------------------------------------
// Main Loop
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY not found in .env")?;
    let client = Client::new();
    let url =
        format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={}", api_key);

    let tools = vec![Tool {
        function_declarations: vec![
            FunctionDeclaration {
                name: "search_library".to_string(),
                description: "Search the database for documents matching a keyword.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": { "keyword": { "type": "STRING" } },
                    "required": ["keyword"]
                })
                ),
            },
            FunctionDeclaration {
                name: "read_toc".to_string(),
                description: "Fetch the root structural nodes (table of contents) of a specific document_id. Use this after finding a relevant document_id.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": { "document_id": { "type": "STRING" } },
                    "required": ["document_id"]
                })
                ),
            },
            FunctionDeclaration {
                name: "drill_down".to_string(),
                description: "Traverse deeper into the document tree by providing a node_id. If the node has children, you will see their summaries. If it is a leaf node, you will get the raw OCR text.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": { "node_id": { "type": "STRING" } },
                    "required": ["node_id"]
                })
                ),
            },
            FunctionDeclaration {
                name: "read_content".to_string(),
                description: "Read the raw OCR text content of a specific leaf node. Use this after drill_down identifies a leaf node whose content you want to read in full.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": { "node_id": { "type": "STRING" } },
                    "required": ["node_id"]
                })
                ),
            },
            FunctionDeclaration {
                name: "resolve_reference".to_string(),
                description: "When the retrieved text mentions a cross-reference (e.g. 'see Appendix G', 'refer to Table 5.3'), use this tool to jump directly to the referenced section. Returns the node_id, title, and summary of the referenced section.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": {
                        "reference_text": { "type": "STRING", "description": "The reference text, e.g. 'Appendix G'" },
                        "document_id": { "type": "STRING", "description": "The document to search within" }
                    },
                    "required": ["reference_text", "document_id"]
                })
                ),
            },
            FunctionDeclaration {
                name: "list_assets".to_string(),
                description: "List images, figures, and tables associated with a document node. Returns asset type, description, page number, and file path for each asset.".to_string(),
                parameters: Some(
                    serde_json::json!({
                    "type": "OBJECT",
                    "properties": { "node_id": { "type": "STRING" } },
                    "required": ["node_id"]
                })
                ),
            }
        ],
    }];

    let mut history: Vec<Content> = vec![
        Content {
            role: "user".to_string(),
            parts: vec![Part {
                text: Some(
                    "You are an autonomous document research agent. Follow this workflow: \
            1. Use search_library to find relevant documents. \
            2. Use read_toc to get the table of contents (root structural nodes). \
            3. Evaluate each section's summary and drill_down into the most promising one. \
            4. After reading a section, ask yourself: 'Do I have enough information to fully answer the question?' \
               - If YES: formulate your final answer. \
               - If NO: return to the table of contents and drill_down into another section. \
            5. You may visit MULTIPLE sections across multiple drill_down calls. Continue until you are confident. \
            6. When you encounter cross-references like 'see Appendix G' or 'refer to Table 5.3', \
               use resolve_reference immediately to jump to that section. \
            7. If you need to understand images, figures, or tables in a section, use list_assets to see what visual elements are available. \
            8. Use read_content to get the full raw text of any leaf node. \
            CRITICAL: NEVER output intermediate conversational text such as 'I found these sections, which one should I read?'. \
            You must independently evaluate the summaries, decide which node is most relevant, and immediately execute the appropriate tool. \
            Only output a final text response to the user AFTER you have finished all tool executions and gathered sufficient information.".to_string()
                ),
                ..Default::default()
            }],
        },
        Content {
            role: "model".to_string(),
            parts: vec![Part {
                text: Some("Understood. I'm ready to research the documents.".to_string()),
                ..Default::default()
            }],
        }
    ];

    println!("===========================================================");
    println!("  Native Rust Agent (gemini-3.1-flash-lite-preview)");
    println!("  Type 'exit' or 'quit' to terminate.");
    println!("===========================================================\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        history.push(Content {
            role: "user".to_string(),
            parts: vec![Part {
                text: Some(input.to_string()),
                ..Default::default()
            }],
        });

        // Loop to handle potential multiple function calls before yielding back to the user
        loop {
            let req_body = GenerateContentRequest {
                contents: history.clone(),
                tools: Some(tools.clone()),
            };

            let res = client.post(&url).json(&req_body).send().await?;
            if !res.status().is_success() {
                println!("! API Error: {}", res.text().await?);
                break;
            }

            let response: GenerateContentResponse = res.json().await?;

            if let Some(mut candidates) = response.candidates {
                if let Some(content) = candidates.pop().and_then(|c| c.content) {
                    history.push(content.clone()); // Append the model's response to history

                    let mut has_func_call = false;
                    let mut current_responses = vec![];

                    for part in content.parts {
                        if let Some(func_call) = part.function_call {
                            has_func_call = true;
                            let result = execute_tool(&func_call.name, &func_call.args);
                            current_responses.push(Part {
                                function_response: Some(FunctionResponse {
                                    name: func_call.name,
                                    response: result,
                                    extra: HashMap::new(),
                                }),
                                ..Default::default()
                            });
                        } else if let Some(text) = part.text {
                            println!("\nAgent: {}", text);
                        }
                    }

                    if has_func_call {
                        // Append the tool results as the next user message and loop again
                        history.push(Content {
                            role: "user".to_string(), // Gemini requires functionResponse to come from the 'function' role, wait, Gemini API maps function responses slightly differently. Let's use "user" role or "function" role depending on the API constraints. Generally, function responses are provided as parts in a Content with the role "user".
                            // It's actually: { role: "function", parts: [...] } or {role: "user", parts: [...]}
                            // For Gemini, it's typically 'user' or 'function'. Let's use 'function'.
                            parts: current_responses,
                        });
                        continue; // Go back and poll Gemini again with the tool results
                    } else {
                        break; // No function calls, model is done.
                    }
                }
            } else {
                println!("\nAgent: [No Response]");
                break;
            }
        }
        println!();
    }

    Ok(())
}
