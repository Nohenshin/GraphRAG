import os
from typing import Dict, Any
from graphrag.utils.logger import logger
from graphrag.utils.config import get_config

class LLMService:
    def generate_answer(self, question: str, context: str, config: Dict[str, Any]) -> str:
        llm_type = config.get("llm_type", "openai")
        api_key = config.get("api_key")
        model = config.get("model", "gpt-3.5-turbo")

        if not context or context.strip() == "":
            logger.warning("Empty context provided to LLM")
            return "I don't have enough information to answer this question."

        try:
            if llm_type == "openai":
                import openai
                openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )
                answer = response.choices[0].message.content
                logger.info(f"OpenAI response generated ({len(answer)} chars)")
                return answer
                
            elif llm_type == "cohere":
                import cohere
                co = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                response = co.generate(
                    model=model,
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.7
                )
                answer = response.generations[0].text.strip()
                logger.info(f"Cohere response generated ({len(answer)} chars)")
                return answer
            else:
                logger.warning(f"Unknown LLM type: {llm_type}")
                return "LLM not configured."
                
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"