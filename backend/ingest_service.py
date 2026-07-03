import os
import tempfile
from graphrag.core.ingest import process_document
from graphrag.core.nlp_graph import process_chunk
from graphrag.core.triplets import process_chunk as process_triplets
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config

class IngestService:
    def __init__(self):
        self.max_tokens = get_config("MAX_TOKENS_PER_CHUNK", 200)
        
    def process_pdf(self, pdf_bytes: bytes, doc_id: str, max_tokens: int = None) -> bool:
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            logger.info(f"Processing PDF: {doc_id}, max_tokens={max_tokens}")
            
            # 1. Ingest: chunk, embed, lưu Neo4j và Qdrant
            chunks, embeddings = process_document(doc_id, pdf_path=tmp_path, max_tokens=max_tokens)
            logger.info(f"Processed {doc_id}: {len(chunks)} chunks, embeddings shape {embeddings.shape}")

            # 2. Trích xuất n-grams và lưu vào Neo4j
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk{idx}"
                process_chunk(chunk_id, chunk_text)
                logger.debug(f"Processed n-grams for chunk {chunk_id}")
                
            # 3. Trích xuất triplets và lưu vào Neo4j
            use_triplets = get_config("USE_TRIPLETS", True)
            if use_triplets:
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk{idx}"
                    triplets = process_triplets(chunk_id, chunk_text)
                    logger.debug(f"Extracted {len(triplets)} triplets for chunk {chunk_id}")

            os.unlink(tmp_path)
            logger.info(f"Successfully ingested {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ingest failed for {doc_id}: {str(e)}")
            return False

    def check_qdrant(self) -> bool:
        try:
            from graphrag.connectors.qdrant_connection import get_connection
            qdrant = get_connection()
            qdrant.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}")
            return False