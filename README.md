# PageIndex Rust (pageindex-rs)

A Rust command-line tool inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) that extracts the document structure of PDF files using Google's Gemini LLM (`gemini-3.1-flash-lite-preview`) and performs high-quality text extraction using a local MLX-powered PaddleOCR service.

## The Architecture: Agentic Tree Search

Unlike traditional RAG systems that rely heavily on vector embeddings (which often hallucinate or lose hierarchical context), `pageindex-rs` implements an **Agentic Tree Search** or "Table of Contents Traversal" pattern.

This system is designed to simulate how a human reads a book:

1. **The Library Level:** The agent surveys the available documents.
2. **The Table of Contents:** The agent reads summary snippets of the main chapters to decide which is relevant.
3. **The Drill-Down:** The agent navigates down into specific sections and only reads the raw text of the exact subsection it needs.

### Storage: SQLite

To support this low-latency, recursive point-lookup traversal pattern, this project abandons heavy vector databases in favor of a local **SQLite** database (`pageindex.db`).

- Every node in the hierarchy is saved with a `summary` (for the LLM to read during traversal).
- Every node explicitly stores a JSON array of its immediate `child_ids` so the LLM always knows if it can drill deeper.
- To save space and processing time, **only leaf nodes** store the actual OCR-extracted text in the `content` column. The Agent can instantly "read" the text of these specific sections without needing to re-parse the PDF.
- **Full-Text Search (FTS5)**: A companion virtual table (`documents_fts`) is automatically synchronized with the database. This allows downstream agents to leverage high-performance, relevance-ranked `MATCH` queries to rapidly locate matching document nodes.

## Implemented Features

- **Direct PDF via API Integration**: Reads local PDF files, encodes them in base64, and sends them directly to the Gemini API as `inlineData` to deduce structural hierarchies.
- **Hybrid Recursive Chunking**:
  - Analyzes long PDFs by breaking them down recursively based on maximum depth limits.
  - LLM Self-Determination: The model flags if a text block contains distinct subdivisions (`has_children: true`).
- **Local Vision-Language Model OCR**: Integrates with a local Python server running `mlx-community/PaddleOCR-VL-1.5-4bit` to perform highly accurate OCR on the leaf nodes. Rust dynamically renders targeted PDF pages into JPEGs using `pdfium-render` and streams them to the local server.
- **SQLite Persistence & Auto-Pruning**:
  - Automatically provisions and populates a relational database capable of serving advanced multi-agent RAG patterns.
  - **Auto-Pruning**: Records the absolute path of the source PDF. If an agent attempts to drill down into a document that has been modified, the database safely self-prunes the entire associated tree to prevent hallucinations.
  - **Modification Tracking**: Computes the SHA-256 hash of the PDF file before extraction. Unmodified files are instantly skipped to save API costs and local compute. Modified files instantly trigger a prune and re-extraction, ensuring perfect synchronization.
- **In-Document Reference Following**: Automatically extracts and resolves internal cross-references (e.g., "see Appendix G"), allowing agents to jump directly to reference targets.
- **Iterative Multi-Section Retrieval**: Features an advanced prompt loop enabling agents to evaluate progress and visit multiple, distinct sections autonomously to synthesize comprehensive answers.
- **Image and Table Extraction**: Detects visual elements during ingestion, rendering them to images and utilizing the OCR service to extract tables as Markdown and generate descriptive captions for images/figures.

## Usage

### Prerequisites

1. Install Rust and Cargo.
2. Install `uv` (Python package manager) for running the local OCR service.
3. Get a Google Gemini API Key.
4. Create a `.env` file in the root of the project with your API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

### 1. Start the Local OCR Service

Before running the Rust extraction tool, you must start the MLX-powered PaddleOCR server. This server requires a Mac with Apple Silicon (for MLX support).

```bash
cd ocr-service
uv run python main.py
```

_(The server will download the model on the first run and bind to `http://0.0.0.0:8080`)_

### 2. Running the Tool (Ingestion)

In a separate terminal, run the tool via Cargo, using the `index` subcommand and providing the path to your PDF. This will parse the document, invoke Gemini to generate the hierarchical json forest, send page images to your local OCR server for text extraction, and save the entire tree to `pageindex.db`.

```bash
# Basic usage with default limits (Depth: 3, Min Pages: 5)
cargo run -- index -p /path/to/your/document.pdf

# Override the hard limits
cargo run -- index --pdf-path /path/to/your/document.pdf --max-depth 4 --min-pages 2

# Override the SQLite database output location (global flag)
cargo run -- --db-url "sqlite:my_custom_index.db?mode=rwc" index -p /path/to/paper.pdf
```

### 3. Querying the Index

You can directly interact with the parsed SQLite database using built-in query subcommands:

**Search for Documents (FTS5):**
Search across the generated summaries and titles for a specific keyword using optimized SQLite FTS5 relevance ranking.

```bash
cargo run -- search "your_keyword"
```

**Get Top-Level Nodes:**
Retrieve the structural root nodes (table of contents) for a specific document ID.
If a top-level node contains further subdivisions, this command will also seamlessly pre-fetch and display the titles and summaries of its immediate child nodes.

```bash
cargo run -- top-nodes "document_uuid_here"
```

**Get Node Details:**
Retrieve the full details (including content preview and children previews) for one or more specific document nodes by their IDs.

```bash
cargo run -- nodes "node_uuid_1" "node_uuid_2"
```

**Read Leaf Node Content:**
Read the raw OCR-extracted text of a specific leaf node.

```bash
cargo run -- read-content "node_uuid"
```

**Resolve Cross-Reference:**
Resolve a cross-reference string (e.g., "Appendix G") to its matching section in the document tree.

```bash
cargo run -- resolve-ref "Appendix G" "document_uuid"
```

**List Node Assets:**
List images, tables, and figures associated with a specific document node.

```bash
cargo run -- list-assets "node_uuid"
```

### Downstream RAG Usage

To consume this data with an Agent, your downstream application should implement simple tools that query the generated `pageindex.db` file:

1. `list_documents()`: `SELECT id, title, overall_summary FROM documents;`
2. `explore_children(node_ids)`: `SELECT node_id, title, summary, has_children, child_ids FROM document_nodes WHERE parent_id IN (?);`
3. `read_content(node_id)`: `SELECT content FROM document_nodes WHERE node_id = ?;`
4. `search_database(keyword)`: `SELECT ... FROM documents_fts WHERE documents_fts MATCH ... ORDER BY rank;`
5. `resolve_reference(text, doc_id)`: `SELECT target_node_id FROM node_references...`
6. `list_assets(node_id)`: `SELECT * FROM node_assets WHERE node_id = ?;`

Alternatively, this project natively defines Agent Skills. For an example of how to plug the search and extraction capabilities into a multi-agent workflow, review `.gemini/skills/pdf-indexer/SKILL.md` and `src/bin/agent.rs`.
