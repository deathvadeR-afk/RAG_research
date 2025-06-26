import os
import logging
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

logger = logging.getLogger(__name__)

# Get database connection from environment or use default
DB_URI = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/research_assistant')

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    title = Column(String(500), nullable=False)
    abstract = Column(Text)
    full_text = Column(Text)
    date_published = Column(DateTime, index=True)
    category = Column(String(50), index=True)
    citation_count = Column(Integer, default=0)
    
    # Relationships
    authors = relationship("PaperAuthor", back_populates="paper")
    topics = relationship("PaperTopic", back_populates="paper")
    citations = relationship("Citation", 
                            foreign_keys="Citation.citing_paper_id",
                            back_populates="citing_paper")

class Author(Base):
    __tablename__ = 'authors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200))
    affiliation = Column(String(300))
    verified = Column(Integer, default=0)
    h_index = Column(Integer)
    
    # Relationships
    papers = relationship("PaperAuthor", back_populates="author")

class PaperAuthor(Base):
    __tablename__ = 'paper_authors'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'))
    author_id = Column(Integer, ForeignKey('authors.id'))
    position = Column(Integer)
    corresponding = Column(Integer, default=0)
    
    # Relationships
    paper = relationship("Paper", back_populates="authors")
    author = relationship("Author", back_populates="papers")

class Citation(Base):
    __tablename__ = 'citations'
    
    id = Column(Integer, primary_key=True)
    citing_paper_id = Column(Integer, ForeignKey('papers.id'))
    cited_paper_id = Column(Integer, ForeignKey('papers.id'))
    context_snippet = Column(Text)
    citation_type = Column(String(50))
    
    # Relationships
    citing_paper = relationship("Paper", foreign_keys=[citing_paper_id], back_populates="citations")
    cited_paper = relationship("Paper", foreign_keys=[cited_paper_id])

class Topic(Base):
    __tablename__ = 'topics'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    parent_topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)
    
    # Relationships
    papers = relationship("PaperTopic", back_populates="topic")
    subtopics = relationship("Topic")

class PaperTopic(Base):
    __tablename__ = 'paper_topics'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'))
    topic_id = Column(Integer, ForeignKey('topics.id'))
    relevance_score = Column(Float)
    
    # Relationships
    paper = relationship("Paper", back_populates="topics")
    topic = relationship("Topic", back_populates="papers")

def setup_database():
    """Create database tables if they don't exist."""
    engine = create_engine(DB_URI)
    logger.info(f"Setting up database with URI: {DB_URI}")
    
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        return engine
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

def get_session():
    """Get a database session."""
    engine = create_engine(DB_URI)
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_database()