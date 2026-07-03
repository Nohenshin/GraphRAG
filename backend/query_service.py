from typing import Dict, Any, Tuple
from graphrag.core.retrieval import (
    hybrid_retrieve,
    hybrid_retrieve_with_triplets,
    retrieve_with_context,
    vector_search,
    GraphRetriever,
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

    def query(self, question: str, config: Dict[str, Any]) -> Tuple[str, str, Dict[str, float]]:
        retrieval_mode = config.get("retrieval_mode", "Hybrid")
        top_k = config.get("top_k", self.default_top_k)
        use_triplets = config.get("use_triplets", True)
        with_context = config.get("with_context", False)
        context_size = config.get("context_size", self.default_context_size)

        logger.info(f"Query: '{question}', mode={retrieval_mode}, top_k={top_k}")

        # 1. Retrieval
        if retrieval_mode == "Vector":
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

        # 2. Build context
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

        # 3. Generate answer
        answer = self.llm.generate_answer(question, context, config)

        # 4. Build graph
        graph_html = self.graph.build_graph(chunks, triplets)

        # 5. Metrics (mock – có thể tích hợp ragas)
        metrics = {
            "context_precision": 0.92,
            "answer_relevancy": 0.87,
            "faithfulness": 0.95,
            "context_recall": 0.88
        }
        
        logger.info("Query completed successfully")
        return answer, graph_html, metrics