use serde::{ Deserialize, Serialize };
use sqlx::{ sqlite::SqlitePool, types::Json };

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct DocumentSummary {
    pub id: String,
    pub title: String,
    pub overall_summary: Option<String>,
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

pub struct LibraryIndex {
    pool: SqlitePool,
}

impl LibraryIndex {
    pub async fn new(db_url: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePool::connect(db_url).await?;
        Ok(Self { pool })
    }

    pub async fn init_tables(&self) -> Result<(), sqlx::Error> {
        let sql =
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                overall_summary TEXT,
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
        Ok(())
    }

    pub async fn insert_document(
        &self,
        id: &str,
        title: &str,
        summary: Option<&str>
    ) -> Result<(), sqlx::Error> {
        sqlx
            ::query("INSERT INTO documents (id, title, overall_summary) VALUES (?, ?, ?)")
            .bind(id)
            .bind(title)
            .bind(summary)
            .execute(&self.pool).await?;
        Ok(())
    }

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
            ::query_as::<_, DocumentSummary>("SELECT id, title, overall_summary FROM documents")
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
    pub async fn read_node_content(&self, node_id: &str) -> Result<NodeContent, sqlx::Error> {
        let content = sqlx
            ::query_as::<_, NodeContent>(
                "SELECT node_id, title, content FROM document_nodes WHERE node_id = ?"
            )
            .bind(node_id)
            .fetch_one(&self.pool).await?;
        Ok(content)
    }
}
