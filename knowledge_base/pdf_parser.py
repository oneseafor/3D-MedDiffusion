"""
PDF Knowledge Base Parser for RAG-based text conditioning.

Extracts pathology descriptions from PDF documents (especially RCM/ARVC
LGE enhancement characteristics) to inject as conditioning signals
into the diffusion model.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PDFKnowledgeParser:
    """Parse PDF documents and extract relevant pathology text."""

    def __init__(
        self,
        pdf_paths: List[str],
        target_keywords: List[str] = None,
        max_text_length: int = 512,
    ):
        self.pdf_paths = [Path(p) for p in pdf_paths]
        self.target_keywords = target_keywords or [
            "LGE", "late gadolinium enhancement", "fibrosis",
            "myocardial", "RCM", "restrictive", "ARVC",
            "arrhythmogenic", "right ventricular",
        ]
        self.max_text_length = max_text_length
        self.knowledge_cache: Dict[str, str] = {}

    def parse_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return ""

        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return ""

        text_blocks = []
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                text_blocks.append(page.get_text())
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to parse {pdf_path}: {e}")
            return ""

        return "\n".join(text_blocks)

    def extract_relevant_sections(self, full_text: str) -> str:
        """Extract sections containing target keywords."""
        if not full_text:
            return ""

        sentences = re.split(r'[.!?\n]+', full_text)
        relevant = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.lower()
            if any(kw.lower() in lower for kw in self.target_keywords):
                relevant.append(sentence)

        # Truncate to max length
        result = " ".join(relevant)
        if len(result) > self.max_text_length:
            result = result[:self.max_text_length]

        return result

    def get_disease_text(self, disease_name: str) -> str:
        """Get knowledge text for a specific disease."""
        # Map disease directory names to search terms
        disease_map = {
            "RCM_niigz": "restrictive cardiomyopathy",
            "ARVC_niigz": "arrhythmogenic right ventricular cardiomyopathy",
            "DCM_niigz": "dilated cardiomyopathy",
            "HCM_niigz": "hypertrophic cardiomyopathy",
            "LVNC_niigz": "left ventricular non-compaction",
        }
        search_term = disease_map.get(disease_name, disease_name)

        # Check cache
        cache_key = disease_name
        if cache_key in self.knowledge_cache:
            return self.knowledge_cache[cache_key]

        # Parse all PDFs and search for disease-specific content
        all_text = []
        for pdf_path in self.pdf_paths:
            raw_text = self.parse_pdf(pdf_path)
            if raw_text:
                all_text.append(raw_text)

        combined = "\n".join(all_text)
        relevant = self.extract_relevant_sections(combined)

        if not relevant:
            # Fallback: generic cardiac description
            relevant = (
                f"Cardiac MRI of {search_term}. "
                "Myocardial tissue characterization using late gadolinium enhancement. "
                "Assessment of myocardial fibrosis and structural abnormalities."
            )

        self.knowledge_cache[cache_key] = relevant
        return relevant

    def get_all_knowledge(self) -> Dict[str, str]:
        """Get knowledge text for all diseases."""
        diseases = ["RCM_niigz", "ARVC_niigz", "DCM_niigz", "HCM_niigz", "LVNC_niigz"]
        return {d: self.get_disease_text(d) for d in diseases}


class TextEncoder(nn.Module):
    """Simple text encoder for conditioning the diffusion model.

    Encodes pathology text descriptions into a fixed-size embedding
    vector that can be injected into the BiFlowNet architecture.
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, max_length: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Simple character-level embedding (no external tokenizer dependency)
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text string to embedding tensor."""
        # Simple character-level tokenization
        chars = [ord(c) % 10000 for c in text[:self.max_length]]
        if not chars:
            chars = [0]
        tokens = torch.tensor(chars, dtype=torch.long)
        return tokens

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: (batch_size, seq_len) token indices
        Returns:
            text_embedding: (batch_size, embed_dim)
        """
        seq_len = text_tokens.shape[1]
        positions = torch.arange(seq_len, device=text_tokens.device).unsqueeze(0)

        x = self.char_embedding(text_tokens) + self.positional_embedding(positions)
        x = self.transformer(x)
        # Mean pooling
        x = x.mean(dim=1)
        x = self.projection(x)
        return x
