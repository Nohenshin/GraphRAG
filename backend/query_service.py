from typing import Dict, Any, Tuple, List
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.core.retrieval import (
    hybrid_retrieve,
    hybrid_retrieve_with_triplets,
    retrieve_with_context,
    vector_search,
    GraphRetriever,
    multi_hop_retrieve,  # import multi-hop
)
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config
from backend.llm_service import LLMService
from backend.graph_service import GraphService


class QueryService:
    def __init__(self):
        self.llm = LLMService()
        self.graph = GraphService()
        self.default_top_k = get_config("TOP_K_RETRIEVAL", 5)
        self.default_context_size = get_config("CONTEXT_SIZE", 2)
        # Load embedding model cho RAGAS metrics
        self.embedder = SentenceTransformer(
            get_config("EMBEDDING_MODEL", "intfloat/e5-base-v2")
        )
        logger.info("QueryService initialized with embedding model for RAGAS")

    def query(self, question: str, config: Dict[str, Any]) -> Tuple[str, str, Dict[str, float]]:
        """Query chính, trả về answer, graph_html, metrics"""
        result = self.query_with_details(question, config)
        return result[0], result[1], result[2]  # answer, graph_html, metrics

    def query_with_details(self, question: str, config: Dict) -> Tuple[str, str, Dict, List]:
        """
        Query đầy đủ chi tiết, trả về:
            - answer (str)
            - graph_html (str)
            - metrics (Dict)
            - chunks (List): danh sách chunks retrieved
        """
        retrieval_mode = config.get("retrieval_mode", "Hybrid")
        top_k = config.get("top_k", self.default_top_k)
        use_triplets = config.get("use_triplets", True)
        with_context = config.get("with_context", False)
        context_size = config.get("context_size", self.default_context_size)
        use_multi_hop = config.get("use_multi_hop", False)
        hops = config.get("hops", 2)

        logger.info(f"Query: '{question}', mode={retrieval_mode}, top_k={top_k}, multi_hop={use_multi_hop}")

        # ==================== 1. RETRIEVAL ====================
        chunks = []
        triplets = []

        if use_multi_hop:
            # Ưu tiên multi-hop nếu được bật
            logger.info(f"Using multi-hop retrieval with hops={hops}")
            chunks = multi_hop_retrieve(question, top_k=top_k, hops=hops)
            # Multi-hop không trả về triplets riêng, nhưng có thể lấy từ graph
            triplets = []

        elif retrieval_mode == "Vector":
            chunks = vector_search(question, top_k=top_k)
            triplets = []

        elif retrieval_mode == "Graph":
            retriever = GraphRetriever()
            chunks = retriever.retrieve_chunks(question, top_k=top_k)
            triplets = []

        else:  # Hybrid
            if with_context:
                results = retrieve_with_context(question, top_k=top_k, context_size=context_size)
                chunks = results
                triplets = []
            elif use_triplets:
                results = hybrid_retrieve_with_triplets(question, top_k=top_k)
                chunks = results.get("chunks", [])
                triplets = results.get("triplets", [])
            else:
                chunks = hybrid_retrieve(question, top_k=top_k)
                triplets = []

        # ==================== 2. BUILD CONTEXT ====================
        if isinstance(chunks, list):
            if chunks and isinstance(chunks[0], dict):
                context_texts = [c.get("text", "") for c in chunks]
            elif chunks and isinstance(chunks[0], tuple):
                context_texts = [c[1] for c in chunks]
            else:
                context_texts = [str(c) for c in chunks]
        else:
            context_texts = []
        context = "\n\n".join(context_texts)

        logger.info(f"Retrieved {len(chunks)} chunks, {len(triplets)} triplets")

        # ==================== 3. GENERATE ANSWER ====================
        answer = self.llm.generate_answer(question, context, config)

        # ==================== 4. BUILD GRAPH ====================
        graph_html = self.graph.build_graph(chunks, triplets)

        # ==================== 5. COMPUTE RAGAS METRICS ====================
        metrics = self._compute_ragas(question, answer, context_texts)

        logger.info("Query completed successfully")
        return answer, graph_html, metrics, chunks

    def _compute_ragas(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Tính các chỉ số RAGAS thực tế bằng embeddings.
        - context_relevancy: trung bình cosine similarity giữa question và các context
        - answer_relevancy: cosine similarity giữa question và answer
        - faithfulness: độ tương đồng trung bình giữa answer và các context
        """
        # Embedding các văn bản
        q_emb = self.embedder.encode(question, normalize_embeddings=True)
        ans_emb = self.embedder.encode(answer, normalize_embeddings=True)
        ctx_embs = self.embedder.encode(contexts, normalize_embeddings=True) if contexts else np.array([])

        # 1. Context Relevancy: trung bình cosine similarity giữa question và từng context
        if len(ctx_embs) > 0:
            sims = cosine_similarity([q_emb], ctx_embs)[0]
            context_relevancy = float(np.mean(sims))
        else:
            context_relevancy = 0.0

        # 2. Answer Relevancy: cosine similarity giữa question và answer
        if ans_emb is not None and len(ans_emb) > 0:
            answer_relevancy = float(cosine_similarity([q_emb], [ans_emb])[0][0])
        else:
            answer_relevancy = 0.0

        # 3. Faithfulness: độ tương đồng trung bình giữa answer và các context
        if len(ctx_embs) > 0:
            ans_ctx_sims = cosine_similarity([ans_emb], ctx_embs)[0]
            faithfulness = float(np.mean(ans_ctx_sims))
        else:
            faithfulness = 0.0

        # Giới hạn trong [0,1]
        return {
            "context_relevancy": max(0.0, min(1.0, context_relevancy)),
            "answer_relevancy": max(0.0, min(1.0, answer_relevancy)),
            "faithfulness": max(0.0, min(1.0, faithfulness))
        }

    def _retrieve(self, question: str, mode: str, top_k: int, use_triplets: bool,
                  with_context: bool, context_size: int) -> Tuple[List, List]:
        """Hàm hỗ trợ retrieval (dùng nội bộ)"""
        if mode == "Vector":
            return vector_search(question, top_k=top_k), []
        elif mode == "Graph":
            retriever = GraphRetriever()
            return retriever.retrieve_chunks(question, top_k=top_k), []
        else:  # Hybrid
            if with_context:
                results = retrieve_with_context(question, top_k=top_k, context_size=context_size)
                return results, []
            elif use_triplets:
                results = hybrid_retrieve_with_triplets(question, top_k=top_k)
                return results.get("chunks", []), results.get("triplets", [])
            else:
                return hybrid_retrieve(question, top_k=top_k), []