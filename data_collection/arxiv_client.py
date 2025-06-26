import arxiv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ArxivClient:
    def __init__(self, max_results: int = 100):
        self.max_results = max_results
        
    def fetch_papers(self, 
                     categories: List[str] = ['cs.AI', 'cs.CL', 'cs.LG'],
                     date_from: Optional[datetime] = None,
                     max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv based on categories and date range."""
        if date_from is None:
            date_from = datetime.now() - timedelta(days=30)
            
        date_str = date_from.strftime('%Y%m%d%H%M%S')
        query = f"cat:({' OR '.join(categories)}) AND submittedDate:[{date_str} TO *]"
        
        logger.info(f"Fetching papers with query: {query}")
        
        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results or self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'categories': result.categories,
                'published': result.published,
                'updated': result.updated,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category
            }
            papers.append(paper)
            
        logger.info(f"Fetched {len(papers)} papers")
        return papers