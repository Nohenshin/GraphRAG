import os
from typing import Dict, Any
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config

class LLMService:
    def generate_answer(self, question: str, context: str, config: Dict[str, Any]) -> str:
        # Chỉ hỗ trợ Cohere, bỏ OpenAI
        api_key = config.get("api_key")
        model = config.get("model", "command-r")  # model mặc định của Cohere

        if not context or context.strip() == "":
            logger.warning("Empty context provided to LLM")
            return "I don't have enough information to answer this question."

        try:
            import cohere
            # Lấy API key từ config hoặc từ biến môi trường
            cohere_api_key = api_key or os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                logger.error("Cohere API key is missing")
                return "Please provide a valid Cohere API key."

            co = cohere.Client(cohere_api_key)
            
            # Sử dụng Chat API mới (thay vì Generate API cũ)
            response = co.chat(
                model=model,
                message=f"Context:\n{context}\n\nQuestion: {question}",
                temperature=0.7,
                max_tokens=300
            )
            
            # Lấy nội dung trả lời
            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                answer = response.message.content
            else:
                # Fallback
                answer = str(response)
                
            logger.info(f"Cohere response generated ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"