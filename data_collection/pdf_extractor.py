import os
import fitz  # PyMuPDF
import requests
import tempfile
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, download_dir: Optional[str] = None):
        self.download_dir = download_dir or tempfile.gettempdir()
        os.makedirs(self.download_dir, exist_ok=True)
        
    def download_pdf(self, pdf_url: str, arxiv_id: str) -> str:
        """Download PDF from URL and save to disk."""
        filepath = os.path.join(self.download_dir, f"{arxiv_id}.pdf")
        
        if os.path.exists(filepath):
            logger.info(f"PDF already exists: {filepath}")
            return filepath
            
        logger.info(f"Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded PDF to {filepath}")
        return filepath
    
    def extract_text(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF."""
        logger.info(f"Extracting text from {pdf_path}")
        
        doc = fitz.open(pdf_path)
        text = ""
        sections = {}
        current_section = "abstract"
        
        # Extract text by page
        for i, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            
            # Simple heuristic for section detection
            if i == 0:
                sections["abstract"] = page_text[:1000]  # Approximate abstract
            
            # Look for common section headers
            for section_name in ["introduction", "methodology", "results", "conclusion", "references"]:
                if section_name.upper() in page_text or section_name.title() in page_text:
                    current_section = section_name
                    sections[current_section] = sections.get(current_section, "") + page_text
        
        # Extract metadata
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "sections": list(sections.keys())
        }
        
        return text, metadata