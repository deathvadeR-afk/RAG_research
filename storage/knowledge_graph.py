import os
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize connection to Neo4j database."""
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        
        logger.info(f"Connecting to Neo4j at {self.uri}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Initialize schema
        self._init_schema()
        
    def _init_schema(self) -> None:
        """Initialize database schema with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE")
            session.run("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT institution_name IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            
            # Create indexes
            session.run("CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)")
            session.run("CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.date_published)")
            session.run("CREATE INDEX author_affiliation IF NOT EXISTS FOR (a:Author) ON (a.affiliation)")
            
            logger.info("Neo4j schema initialized with constraints and indexes")
    
    def add_paper(self, paper: Dict[str, Any]) -> None:
        """Add a paper and its relationships to the knowledge graph."""
        with self.driver.session() as session:
            # Create paper node
            session.run("""
                MERGE (p:Paper {arxiv_id: $arxiv_id})
                ON CREATE SET 
                    p.title = $title,
                    p.abstract = $abstract,
                    p.date_published = $date_published,
                    p.category = $primary_category
                ON MATCH SET
                    p.title = $title,
                    p.abstract = $abstract,
                    p.date_published = $date_published,
                    p.category = $primary_category
            """, {
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'date_published': paper['publication_date'],
                'primary_category': paper['primary_category']
            })
            
            # Create authors and relationships
            for author in paper.get('authors', []):
                session.run("""
                    MERGE (a:Author {name: $name})
                    ON CREATE SET a.affiliation = $affiliation
                    WITH a
                    MATCH (p:Paper {arxiv_id: $arxiv_id})
                    MERGE (a)-[:AUTHORED]->(p)
                """, {
                    'name': author['full_name'],
                    'affiliation': author.get('affiliation', ''),
                    'arxiv_id': paper['arxiv_id']
                })
                
                # If affiliation is known, create institution node
                if author.get('affiliation'):
                    session.run("""
                        MERGE (i:Institution {name: $institution})
                        WITH i
                        MATCH (a:Author {name: $author_name})
                        MERGE (a)-[:BELONGS_TO]->(i)
                    """, {
                        'institution': author['affiliation'],
                        'author_name': author['full_name']
                    })
            
            # Create topic relationships
            for category in paper.get('categories', []):
                session.run("""
                    MERGE (t:Topic {name: $category})
                    WITH t
                    MATCH (p:Paper {arxiv_id: $arxiv_id})
                    MERGE (p)-[:CATEGORIZED_AS]->(t)
                """, {
                    'category': category,
                    'arxiv_id': paper['arxiv_id']
                })
                
            logger.info(f"Added paper {paper['arxiv_id']} to knowledge graph")
    
    def add_citation(self, citing_paper_id: str, cited_paper_id: str, context: Optional[str] = None) -> None:
        """Add citation relationship between papers."""
        with self.driver.session() as session:
            session.run("""
                MATCH (citing:Paper {arxiv_id: $citing_id})
                MATCH (cited:Paper {arxiv_id: $cited_id})
                MERGE (citing)-[c:CITES]->(cited)
                ON CREATE SET c.context = $context
            """, {
                'citing_id': citing_paper_id,
                'cited_id': cited_paper_id,
                'context': context or ''
            })
            
            logger.info(f"Added citation from {citing_paper_id} to {cited_paper_id}")
    
    def query_related_papers(self, paper_id: str, max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers related to the given paper through citations and common topics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Paper {arxiv_id: $paper_id})
                OPTIONAL MATCH path1 = (p)-[:CITES*1..2]->(cited:Paper)
                OPTIONAL MATCH path2 = (p)<-[:CITES*1..2]-(citing:Paper)
                OPTIONAL MATCH (p)-[:CATEGORIZED_AS]->(t:Topic)<-[:CATEGORIZED_AS]-(related:Paper)
                WHERE related <> p
                WITH collect(distinct cited) + collect(distinct citing) + collect(distinct related) as papers
                UNWIND papers as paper
                RETURN DISTINCT paper.arxiv_id as arxiv_id, 
                       paper.title as title,
                       paper.abstract as abstract,
                       paper.date_published as date_published,
                       paper.category as category
                LIMIT $limit
            """, {
                'paper_id': paper_id,
                'limit': limit
            })
            
            return [dict(record['paper']) for record in result]
    
    def query_author_collaborators(self, author_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find collaborators of the given author."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Author {name: $author_name})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(collaborator:Author)
                WHERE collaborator <> a
                WITH collaborator, count(p) as collaboration_count
                RETURN collaborator.name as name, 
                       collaborator.affiliation as affiliation,
                       collaboration_count
                ORDER BY collaboration_count DESC
                LIMIT $limit
            """, {
                'author_name': author_name,
                'limit': limit
            })
            
            return [dict(record) for record in result]
    
    def close(self) -> None:
        """Close the database connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")