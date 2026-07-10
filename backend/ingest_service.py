import os
import tempfile
from typing import Tuple, Dict, Any, List, Callable, Optional
from graphrag.core.ingest import process_document
from graphrag.core.nlp_graph import process_chunk as process_nlp_chunk
from graphrag.core.triplets import process_chunk as process_triplets
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config

class IngestService:
    def __init__(self):
        self.max_tokens = get_config("MAX_TOKENS_PER_CHUNK", 200)

    def process_pdf_with_progress(
        self,
        pdf_bytes: bytes,
        doc_id: str,
        max_tokens: int = None,
        progress_callback: Optional[Callable[[str, str, bool, bool], None]] = None
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Xử lý PDF và gọi callback để cập nhật tiến trình.
        progress_callback: hàm nhận (step_name, message, is_complete, is_error)
        Returns: (success, info_dict, status_dict)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        status = {
            "chunking": {"status": "pending", "message": ""},
            "embedding": {"status": "pending", "message": ""},
            "neo4j": {"status": "pending", "message": ""},
            "qdrant": {"status": "pending", "message": ""},
            "graph": {"status": "pending", "message": ""},
            "triplets": {"status": "skipped", "message": ""},
        }
        info = {}

        def update(step: str, message: str, is_complete: bool = False, is_error: bool = False):
            status[step]["message"] = message
            if is_error:
                status[step]["status"] = "error"
            elif is_complete:
                status[step]["status"] = "done"
            else:
                status[step]["status"] = "running"
            if progress_callback:
                progress_callback(step, message, is_complete, is_error)

        try:
            # 1. Lưu PDF tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # 2. Chunking
            update("chunking", "Đang chunking văn bản...", is_complete=False)
            chunks, embeddings = process_document(doc_id, pdf_path=tmp_path, max_tokens=max_tokens)
            update("chunking", f"✅ Đã tạo {len(chunks)} chunks", is_complete=True)
            info["num_chunks"] = len(chunks)
            info["chunks"] = chunks[:5]
            info["embed_dim"] = embeddings.shape[1] if len(embeddings) > 0 else 0

            # 3. Embedding (đã có từ process_document)
            update("embedding", f"✅ Đã tạo embeddings (dim={info['embed_dim']})", is_complete=True)

            # 4. Lưu Neo4j (chunk + term graph)
            update("neo4j", "Đang lưu chunks và term graph vào Neo4j...", is_complete=False)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk{idx}"
                process_nlp_chunk(chunk_id, chunk_text)
            update("neo4j", "✅ Đã lưu chunks và term graph vào Neo4j", is_complete=True)

            # 5. Lưu Qdrant
            update("qdrant", f"Đang lưu {len(chunks)} embeddings vào Qdrant...", is_complete=False)
            from graphrag.connectors.qdrant_connection import get_connection as get_qdrant
            qdrant = get_qdrant()
            ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
            metadata = [{"doc_id": doc_id, "chunk_index": i, "text": chunk[:1000]} for i, chunk in enumerate(chunks)]
            result = qdrant.upsert_vectors(
                collection_name="tokens",
                vectors=embeddings.tolist(),
                ids=ids,
                metadata=metadata,
                timeout=120  # tăng timeout lên 2 phút
            )
            if result:
                update("qdrant", f"✅ Đã lưu {len(chunks)} embeddings vào Qdrant", is_complete=True)
            else:
                update("qdrant", "❌ Lưu Qdrant thất bại", is_error=True)

            # 6. Graph
            update("graph", "✅ Đồ thị term đã sẵn sàng", is_complete=True)

            # 7. Triplets (nếu bật)
            use_triplets = get_config("USE_TRIPLETS", True)
            if use_triplets:
                update("triplets", "Đang trích xuất triplets...", is_complete=False)
                # Triplets đã được xử lý trong process_document, nhưng ta vẫn gọi lại để đảm bảo
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk{idx}"
                    process_triplets(chunk_id, chunk_text)
                update("triplets", "✅ Đã trích xuất triplets", is_complete=True)
            else:
                update("triplets", "⏭️ Bỏ qua triplets (đã tắt)", is_complete=True)

            os.unlink(tmp_path)
            return True, info, status

        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            # Đánh dấu bước đang chạy bị lỗi
            for step in status:
                if status[step]["status"] == "running":
                    status[step]["status"] = "error"
                    status[step]["message"] = f"❌ Lỗi: {str(e)[:100]}"
            return False, info, status

    # ==================== Hàm cũ giữ tương thích ====================
    def process_pdf(self, pdf_bytes: bytes, doc_id: str, max_tokens: int = None) -> bool:
        """Giữ nguyên cho tương thích với code cũ"""
        success, _, _ = self.process_pdf_with_progress(pdf_bytes, doc_id, max_tokens)
        return success

    def process_pdf_with_info(self, pdf_bytes: bytes, doc_id: str, max_tokens: int = None):
        """Giữ nguyên cho tương thích với code cũ, trả về (success, info_dict)"""
        success, info, _ = self.process_pdf_with_progress(pdf_bytes, doc_id, max_tokens)
        return success, info

    def check_qdrant(self) -> bool:
        try:
            from graphrag.connectors.qdrant_connection import get_connection
            qdrant = get_connection()
            qdrant.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}")
            return False