use serde::{ Deserialize, Serialize };
use sqlx::{ sqlite::SqlitePool, types::Json };

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct DocumentSummary {
    pub id: String,
    pub title: String,
    pub overall_summary: Option<String>,
    pub file_path: Option<String>,
    pub file_hash: Option<String>,
}

#[allow(dead_code)]
pub enum HashCheckResult {
    Match,
    Mismatch(String), // Returns the old document ID so we can prune it
    NotFound,
}

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct TraversalNode {
    pub node_id: String,
    pub title: String,
    pub summary: String,
    pub has_children: bool,
    pub child_ids: Json<Vec<String>>,
}

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct NodeContent {
    pub node_id: String,
    pub title: String,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct FullDocumentNode {
    pub node_id: String,
    pub document_id: String,
    pub parent_id: Option<String>,
    pub title: String,
    pub summary: String,
    pub content: Option<String>,
    pub start_index: Option<i32>,
    pub end_index: Option<i32>,
    pub has_children: bool,
    pub child_ids: Json<Vec<String>>,
}

pub struct LibraryIndex {
    pool: SqlitePool,
}

impl LibraryIndex {
    pub async fn new(db_url: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePool::connect(db_url).await?;
        Ok(Self { pool })
    }

    #[allow(dead_code)]
    pub async fn init_tables(&self) -> Result<(), sqlx::Error> {
        let sql =
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                overall_summary TEXT,
                file_path TEXT,
                file_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_nodes (
                node_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_id TEXT REFERENCES document_nodes(node_id) ON DELETE CASCADE,
                
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT,
                
                start_index INTEGER,
                end_index INTEGER,
                has_children BOOLEAN NOT NULL,
                child_ids JSON NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_doc ON document_nodes(document_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON document_nodes(parent_id);
        "#;
        sqlx::query(sql).execute(&self.pool).await?;
        // 3. Create FTS5 virtual table for lightning-fast, relevance-ranked searching
        sqlx
            ::query(
                r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title,
                overall_summary,
                content='documents',
                content_rowid='rowid'
            );
            "#
            )
            .execute(&self.pool).await?;

        // 4. Create triggers to automatically keep FTS5 synchronized
        sqlx
            ::query(
                r#"
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
              INSERT INTO documents_fts(rowid, title, overall_summary) VALUES (new.rowid, new.title, new.overall_summary);
            END;
            "#
            )
            .execute(&self.pool).await?;

        sqlx
            ::query(
                r#"
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
              INSERT INTO documents_fts(documents_fts, rowid, title, overall_summary) VALUES ('delete', old.rowid, old.title, old.overall_summary);
            END;
            "#
            )
            .execute(&self.pool).await?;

        sqlx
            ::query(
                r#"
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
              INSERT INTO documents_fts(documents_fts, rowid, title, overall_summary) VALUES ('delete', old.rowid, old.title, old.overall_summary);
              INSERT INTO documents_fts(rowid, title, overall_summary) VALUES (new.rowid, new.title, new.overall_summary);
            END;
            "#
            )
            .execute(&self.pool).await?;

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn insert_document(
        &self,
        id: &str,
        title: &str,
        summary: Option<&str>,
        file_path: Option<&str>,
        file_hash: Option<&str>
    ) -> Result<(), sqlx::Error> {
        sqlx
            ::query(
                "INSERT INTO documents (id, title, overall_summary, file_path, file_hash) VALUES (?, ?, ?, ?, ?)"
            )
            .bind(id)
            .bind(title)
            .bind(summary)
            .bind(file_path)
            .bind(file_hash)
            .execute(&self.pool).await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn check_document_hash(
        &self,
        absolute_file_path: &str,
        new_hash: &str
    ) -> Result<HashCheckResult, sqlx::Error> {
        let record = sqlx
            ::query_as::<_, (String, Option<String>)>(
                "SELECT id, file_hash FROM documents WHERE file_path = ?"
            )
            .bind(absolute_file_path)
            .fetch_optional(&self.pool).await?;

        match record {
            Some((id, file_hash)) => {
                if file_hash.as_deref() == Some(new_hash) {
                    Ok(HashCheckResult::Match)
                } else {
                    Ok(HashCheckResult::Mismatch(id))
                }
            }
            None => Ok(HashCheckResult::NotFound),
        }
    }

    #[allow(dead_code)]
    pub async fn prune_document(&self, doc_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM documents WHERE id = ?").bind(doc_id).execute(&self.pool).await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn insert_node(
        &self,
        node_id: &str,
        document_id: &str,
        parent_id: Option<&str>,
        title: &str,
        summary: &str,
        content: Option<&str>,
        start_index: i32,
        end_index: i32,
        has_children: bool,
        child_ids: &[String]
    ) -> Result<(), sqlx::Error> {
        let child_ids_json = serde_json::to_string(child_ids).unwrap_or_else(|_| "[]".to_string());
        sqlx
            ::query(
                "INSERT INTO document_nodes (node_id, document_id, parent_id, title, summary, content, start_index, end_index, has_children, child_ids)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(node_id)
            .bind(document_id)
            .bind(parent_id)
            .bind(title)
            .bind(summary)
            .bind(content)
            .bind(start_index)
            .bind(end_index)
            .bind(has_children)
            .bind(child_ids_json)
            .execute(&self.pool).await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn list_documents(&self) -> Result<Vec<DocumentSummary>, sqlx::Error> {
        let docs = sqlx
            ::query_as::<_, DocumentSummary>(
                "SELECT id, title, overall_summary, file_path, file_hash FROM documents"
            )
            .fetch_all(&self.pool).await?;
        Ok(docs)
    }

    #[allow(dead_code)]
    pub async fn search_documents_by_summary(
        &self,
        keyword: &str
    ) -> Result<Vec<DocumentSummary>, sqlx::Error> {
        // Basic FTS MATCH query
        let docs = sqlx
            ::query_as::<_, DocumentSummary>(
                r#"
        SELECT d.id, d.title, d.overall_summary, d.file_path, d.file_hash 
        FROM documents d 
        JOIN documents_fts f ON d.rowid = f.rowid 
        WHERE documents_fts MATCH ? 
        ORDER BY f.rank
        "#
            )
            .bind(keyword)
            .fetch_all(&self.pool).await?;
        Ok(docs)
    }
    #[allow(dead_code)]
    pub async fn get_top_level_nodes(
        &self,
        document_ids: &[String]
    ) -> Result<Vec<TraversalNode>, sqlx::Error> {
        if document_ids.is_empty() {
            return Ok(vec![]);
        }
        let ids_str = document_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let query_str =
            format!("SELECT node_id, title, summary, has_children, child_ids 
             FROM document_nodes 
             WHERE parent_id IS NULL AND document_id IN ({})", ids_str);
        let mut query = sqlx::query_as::<_, TraversalNode>(&query_str);
        for id in document_ids {
            query = query.bind(id);
        }
        let nodes = query.fetch_all(&self.pool).await?;
        Ok(nodes)
    }

    #[allow(dead_code)]
    pub async fn explore_children(
        &self,
        node_ids: &[String]
    ) -> Result<Vec<TraversalNode>, sqlx::Error> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }
        let ids_str = node_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let query_str =
            format!("SELECT node_id, title, summary, has_children, child_ids 
             FROM document_nodes 
             WHERE parent_id IN ({})", ids_str);
        let mut query = sqlx::query_as::<_, TraversalNode>(&query_str);
        for id in node_ids {
            query = query.bind(id);
        }
        let child_nodes = query.fetch_all(&self.pool).await?;
        Ok(child_nodes)
    }

    #[allow(dead_code)]
    pub async fn get_nodes_by_ids(
        &self,
        node_ids: &[String]
    ) -> Result<Vec<TraversalNode>, sqlx::Error> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }
        let ids_str = node_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let query_str =
            format!("SELECT node_id, title, summary, has_children, child_ids 
             FROM document_nodes 
             WHERE node_id IN ({})", ids_str);
        let mut query = sqlx::query_as::<_, TraversalNode>(&query_str);
        for id in node_ids {
            query = query.bind(id);
        }
        let nodes = query.fetch_all(&self.pool).await?;
        Ok(nodes)
    }

    #[allow(dead_code)]
    pub async fn get_full_nodes_by_ids(
        &self,
        node_ids: &[String]
    ) -> Result<Vec<FullDocumentNode>, sqlx::Error> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }
        let ids_str = node_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let query_str =
            format!("SELECT node_id, document_id, parent_id, title, summary, content, start_index, end_index, has_children, child_ids 
             FROM document_nodes 
             WHERE node_id IN ({})", ids_str);
        let mut query = sqlx::query_as::<_, FullDocumentNode>(&query_str);
        for id in node_ids {
            query = query.bind(id);
        }
        let nodes = query.fetch_all(&self.pool).await?;
        Ok(nodes)
    }

    #[allow(dead_code)]
    pub async fn read_node_content(
        &self,
        node_id: &str
    ) -> Result<Option<NodeContent>, anyhow::Error> {
        #[derive(sqlx::FromRow)]
        struct JoinedNode {
            doc_id: String,
            file_path: Option<String>,
            node_id: String,
            title: String,
            content: Option<String>,
        }

        // First check if the file still exists by joining back to the documents table
        let record = sqlx
            ::query_as::<_, JoinedNode>(
                "SELECT d.id as doc_id, d.file_path, n.node_id, n.title, n.content 
             FROM document_nodes n 
             JOIN documents d ON n.document_id = d.id 
             WHERE n.node_id = ?"
            )
            .bind(node_id)
            .fetch_optional(&self.pool).await?;

        if let Some(row) = record {
            if let Some(file_path_str) = row.file_path {
                let file_path = std::path::Path::new(&file_path_str);

                // If the file is missing, prune the document from the database (CASCADE deletes nodes)
                if !file_path.exists() {
                    println!(
                        "Warning: Associated PDF file {} is missing. Pruning document {} from database.",
                        file_path_str,
                        row.doc_id
                    );
                    sqlx
                        ::query("DELETE FROM documents WHERE id = ?")
                        .bind(row.doc_id)
                        .execute(&self.pool).await?;

                    return Ok(None);
                }
            }

            return Ok(
                Some(NodeContent {
                    node_id: row.node_id,
                    title: row.title,
                    content: row.content,
                })
            );
        }

        Ok(None)
    }
}
