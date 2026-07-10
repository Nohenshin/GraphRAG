"""
Document ingestion and processing utilities for GraphRAG
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import nltk

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from graphrag.connectors.neo4j_connection import get_connection as get_neo4j_connection
from graphrag.connectors.qdrant_connection import get_connection as get_qdrant_connection
from graphrag.utils.common import embed_text, DEFAULT_EMBEDDING_MODEL
from graphrag.utils.logger import logger

# No need to download NLTK resources here as it's handled in __init__.py

class DocumentIngestor:
    """Handles document loading, processing, and storage in Neo4j and Qdrant"""
    
    def __init__(self, neo4j_conn=None, qdrant_conn=None, embedding_model=DEFAULT_EMBEDDING_MODEL):
        self.neo4j = neo4j_conn or get_neo4j_connection()
        self.qdrant = qdrant_conn or get_qdrant_connection()
        self.embedding_model = embedding_model
        logger.info(f"Using embedding model: {embedding_model}")
        
    def load_pdf(self, path: str) -> str:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required to load PDF files. Install with 'pip install pymupdf'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
        logger.info(f"Loading PDF from {path}")
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            logger.info(f"Successfully extracted text from PDF: {path}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {path}: {str(e)}")
            raise
            
    def chunk_text(self, text: str, max_tokens: int = 200) -> List[str]:
        if not text:
            logger.warning("Received empty text for chunking")
            return []
        logger.info(f"Chunking text ({len(text)} chars) with max {max_tokens} tokens per chunk")
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            if current_length + len(tokens) > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sent)
            current_length += len(tokens)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        logger.info(f"Created {len(chunks)} chunks from input text")
        return chunks
        
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return np.array([])
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            embeddings = embed_text(chunks, model_name=self.embedding_model, normalize=True)
            logger.info(f"Successfully generated embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
            
    def store_chunks_in_neo4j(self, doc_id: str, chunks: List[str], 
                          embeddings: np.ndarray) -> None:
        doc_query = """
        MERGE (d:Document {id: $doc_id})
        RETURN d
        """
        self.neo4j.run_query(doc_query, {"doc_id": doc_id})
        prev_chunk_id = None
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk{i}"
            chunk_query = """
            MATCH (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, 
                c.index = $index
            MERGE (d)-[:CONTAINS]->(c)
            RETURN c
            """
            params = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "index": i
            }
            self.neo4j.run_query(chunk_query, params)
            if prev_chunk_id is not None:
                relationship_query = """
                MATCH (prev:Chunk {id: $prev_chunk_id})
                WITH prev
                MATCH (curr:Chunk {id: $curr_chunk_id})
                MERGE (prev)-[:NEXT]->(curr)
                MERGE (curr)-[:PREV]->(prev)
                """
                relationship_params = {
                    "prev_chunk_id": prev_chunk_id,
                    "curr_chunk_id": chunk_id
                }
                self.neo4j.run_query(relationship_query, relationship_params)
            prev_chunk_id = chunk_id
        logger.info(f"Stored {len(chunks)} chunks for document {doc_id} in Neo4j with NEXT/PREV relationships")
        
    def store_embeddings_in_qdrant(self, doc_id: str, chunks: List[str], 
                                embeddings: np.ndarray) -> None:
        ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
        metadata = [{"doc_id": doc_id, "chunk_index": i, "text": chunk[:1000]} for i, chunk in enumerate(chunks)]
        result = self.qdrant.upsert_vectors(
            collection_name="tokens", 
            vectors=embeddings.tolist(), 
            ids=ids, 
            metadata=metadata,
            timeout=120
        )
        if result:
            logger.info(f"Stored {len(chunks)} embeddings for document {doc_id} in Qdrant")
        else:
            logger.error(f"Failed to store embeddings for document {doc_id} in Qdrant")
            
    def process_document(self, doc_id: str, text: str = None, pdf_path: str = None, 
                     max_tokens: int = 200) -> Tuple[List[str], np.ndarray]:
        if text is None and pdf_path is not None:
            text = self.load_pdf(pdf_path)
        elif text is None:
            raise ValueError("Either text or pdf_path must be provided")
        chunks = self.chunk_text(text, max_tokens)
        logger.info(f"Split document into {len(chunks)} chunks")
        embeddings = self.embed_chunks(chunks)
        logger.info(f"Generated embeddings of shape {embeddings.shape}")
        self.store_chunks_in_neo4j(doc_id, chunks, embeddings)
        self.store_embeddings_in_qdrant(doc_id, chunks, embeddings)
        return chunks, embeddings


# ==================== CONVENIENCE FUNCTIONS ====================

def process_document(doc_id: str, text: str = None, pdf_path: str = None, 
                 max_tokens: int = 200) -> Tuple[List[str], np.ndarray]:
    ingestor = DocumentIngestor()
    return ingestor.process_document(doc_id, text, pdf_path, max_tokens)

def load_pdf(path: str) -> str:
    ingestor = DocumentIngestor()
    return ingestor.load_pdf(path)

def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    ingestor = DocumentIngestor()
    return ingestor.chunk_text(text, max_tokens)

def embed_chunks(chunks: List[str]) -> np.ndarray:
    ingestor = DocumentIngestor()
    return ingestor.embed_chunks(chunks)

def store_chunks_in_neo4j(doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
    ingestor = DocumentIngestor()
    ingestor.store_chunks_in_neo4j(doc_id, chunks, embeddings)

def store_embeddings_in_qdrant(doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
    ingestor = DocumentIngestor()
    ingestor.store_embeddings_in_qdrant(doc_id, chunks, embeddings)