# PageIndex Rust (pageindex-rs)

A Rust command-line tool inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) that extracts the document structure and content of PDF files using Google's Gemini LLM (`gemini-3.1-flash-lite-preview`).

## The Architecture: Agentic Tree Search

Unlike traditional RAG systems that rely heavily on vector embeddings (which often hallucinate or lose hierarchical context), `pageindex-rs` implements an **Agentic Tree Search** or "Table of Contents Traversal" pattern.

This system is designed to simulate how a human reads a book:

1. **The Library Level:** The agent surveys the available documents.
2. **The Table of Contents:** The agent reads summary snippets of the main chapters to decide which is relevant.
3. **The Drill-Down:** The agent navigates down into specific sections and only reads the raw text of the exact subsection it needs.

### Storage: SQLite

To support this low-latency, recursive point-lookup traversal pattern, this project abandons heavy vector databases in favor of a local **SQLite** database (`pageindex.db`).

- Every node in the hierarchy is saved with a `summary` (for the LLM to read during traversal).
- Every node explicitely stores a JSON array of its immediate `child_ids` so the LLM always knows if it can drill deeper.
- The actual page content is extracted during the ingestion phase and stored in the `content` column, meaning the Agent can instantly "read" the text without re-parsing the PDF.

## Implemented Features

- **Direct PDF via API Integration**: Reads local PDF files, encodes them in base64, and sends them directly to the Gemini API as `inlineData`.
- **Hybrid Recursive Chunking**:
  - Analyzes long PDFs by breaking them down recursively based on maximum depth limits.
  - LLM Self-Determination: The model flags if a text block contains distinct subdivisions (`has_children: true`).
- **In-Memory PDF Slicing with Text Extraction**: Uses the `lopdf` crate to dynamically slice massive PDFs into targeted byte blocks for LLM structure reasoning, while simultaneously extracting the raw text for the SQLite database.
- **SQLite Persistence**: Automatically provisions and populates a relational database capable of serving advanced multi-agent RAG patterns.

## Usage

### Prerequisites

1. Install Rust and Cargo.
2. Get a Google Gemini API Key.
3. Create a `.env` file in the root of the project with your API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

### Running the Tool (Ingestion)

Run the tool via Cargo, providing the path to your PDF. This will parse the document, invoke Gemini to generate the hierarchical json forest, extract the text, and save the entire tree to `pageindex.db`.

```bash
# Basic usage with default limits (Depth: 3, Min Pages: 5)
cargo run -- --pdf-path /path/to/your/document.pdf

# Override the hard limits
cargo run -- --pdf-path /path/to/your/document.pdf --max-depth 4 --min-pages 2

# Override the SQLite database output location
cargo run -- --pdf-path /path/to/paper.pdf --db-url "sqlite:my_custom_index.db?mode=rwc"
```

### Downstream RAG Usage

To consume this data with an Agent, your downstream application should implement three simple tools that query the generated `pageindex.db` file:

1. `list_documents()`: `SELECT id, title, overall_summary FROM documents;`
2. `explore_children(node_ids)`: `SELECT node_id, title, summary, has_children, child_ids FROM document_nodes WHERE parent_id IN (?);`
3. `read_content(node_id)`: `SELECT content FROM document_nodes WHERE node_id = ?;`
