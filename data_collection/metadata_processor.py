import re
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataProcessor:
    def __init__(self):
        self.institution_patterns = [
            r'University of ([A-Za-z\s]+)',
            r'([A-Za-z\s]+) University',
            r'([A-Za-z\s]+) Institute of Technology',
            r'([A-Za-z\s]+) College'
        ]
    
    def normalize_paper(self, paper: Dict[str, Any], full_text: str = "") -> Dict[str, Any]:
        """Normalize and enrich paper metadata."""
        normalized = {
            'arxiv_id': paper['arxiv_id'],
            'title': self._normalize_title(paper['title']),
            'abstract': self._clean_text(paper['abstract']),
            'authors': self._normalize_authors(paper['authors']),
            'publication_date': paper['published'].strftime('%Y-%m-%d'),
            'update_date': paper['updated'].strftime('%Y-%m-%d'),
            'categories': paper['categories'],
            'primary_category': paper['primary_category'],
            'pdf_url': paper['pdf_url'],
            'processed_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Extract additional metadata if full text is available
        if full_text:
            normalized.update({
                'institutions': self._extract_institutions(full_text),
                'citation_count': None,  # To be populated later
                'references': self._extract_references(full_text)
            })
            
        return normalized
    
    def _normalize_title(self, title: str) -> str:
        """Clean and normalize paper title."""
        # Remove newlines and excessive spaces
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    
    def _normalize_authors(self, authors: List[str]) -> List[Dict[str, Any]]:
        """Normalize author information."""
        normalized_authors = []
        for author in authors:
            # Split name into parts (simple approach)
            parts = author.split()
            if len(parts) > 1:
                normalized_authors.append({
                    'full_name': author,
                    'last_name': parts[-1],
                    'first_name': ' '.join(parts[:-1]),
                    'affiliation': None  # To be populated later
                })
            else:
                normalized_authors.append({
                    'full_name': author,
                    'last_name': author,
                    'first_name': '',
                    'affiliation': None
                })
        return normalized_authors
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_institutions(self, text: str) -> List[str]:
        """Extract institution names from text."""
        institutions = []
        for pattern in self.institution_patterns:
            matches = re.findall(pattern, text)
            institutions.extend(matches)
        return list(set(institutions))
    
    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references from paper text."""
        # Simple heuristic: look for reference section and extract entries
        references = []
        ref_section_match = re.search(r'References(.*?)(?:$|Appendix)', text, re.DOTALL | re.IGNORECASE)
        
        if ref_section_match:
            ref_text = ref_section_match.group(1)
            # Look for patterns like [1] Author, Title...
            ref_entries = re.findall(r'\[\d+\](.*?)(?=\[\d+\]|$)', ref_text, re.DOTALL)
            
            for entry in ref_entries:
                entry = entry.strip()
                if entry:
                    references.append({
                        'text': entry,
                        'arxiv_id': self._extract_arxiv_id(entry),
                        'doi': self._extract_doi(entry)
                    })
        
        return references
    
    def _extract_arxiv_id(self, text: str) -> str:
        """Extract arXiv ID from text if present."""
        match = re.search(r'arxiv:(\d+\.\d+)', text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_doi(self, text: str) -> str:
        """Extract DOI from text if present."""
        match = re.search(r'doi:([^\s]+)', text, re.IGNORECASE)
        return match.group(1) if match else None