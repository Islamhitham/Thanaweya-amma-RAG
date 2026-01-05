"""
Ollama LLM client for question answering, quiz generation, and explanations
"""
import ollama
from typing import List, Dict, Optional
from . import config


class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self):
        """Initialize Ollama client"""
        self.model = config.OLLAMA_MODEL
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
        
        # Verify model is available
        try:
            self.client.list()
            print(f"Connected to Ollama. Using model: {self.model}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def _create_context_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        mode: str,
        conversation_history: str = ""
    ) -> str:
        """
        Create prompt with context and conversation history
        
        Args:
            query: User query
            context_docs: Retrieved documents for context
            mode: 'ask', 'quiz', or 'explain'
            conversation_history: Formatted conversation history
            
        Returns:
            Complete prompt string
        """
        # Build context from retrieved documents
        context_text = "\n\n".join([
            f"[مصدر {i+1}]\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Select system prompt based on mode
        system_prompts = {
            "ask": config.SYSTEM_PROMPT_QA,
            "quiz": config.SYSTEM_PROMPT_QUIZ,
            "explain": config.SYSTEM_PROMPT_EXPLAIN
        }
        system_prompt = system_prompts.get(mode, config.SYSTEM_PROMPT_QA)
        
        # Build full prompt
        prompt_parts = [system_prompt]
        
        if conversation_history:
            prompt_parts.append(f"\n--- Previous Conversation ---\n{conversation_history}")
        
        if context_text:
            prompt_parts.append(f"\n--- Reference Content ---\n{context_text}")
        
        # Add specific instructions based on mode
        if mode == "quiz":
            prompt_parts.append(
                "\n--- Instructions ---\n"
                "قم بإنشاء 5 أسئلة اختيار من متعدد بناءً على المحتوى أعلاه.\n"
                "لكل سؤال، قدم 4 خيارات (A, B, C, D) واذكر الإجابة الصحيحة.\n\n"
                "Create 5 multiple choice questions based on the content above.\n"
                "For each question, provide 4 options (A, B, C, D) and indicate the correct answer."
            )
        elif mode == "explain":
            prompt_parts.append(
                f"\n--- Topic to Explain ---\n{query}\n\n"
                "اشرح هذا الموضوع بطريقة واضحة ومبسطة مع أمثلة.\n"
                "Explain this topic clearly and simply with examples."
            )
        else:  # ask mode
            prompt_parts.append(f"\n--- Student Question ---\n{query}")
        
        return "\n".join(prompt_parts)
    
    def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        mode: str = "ask",
        conversation_history: str = "",
        stream: bool = False
    ) -> str:
        """
        Generate response using Ollama
        
        Args:
            query: User query
            context_docs: Retrieved documents for context
            mode: 'ask', 'quiz', or 'explain'
            conversation_history: Previous conversation context
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        # Create prompt
        prompt = self._create_context_prompt(
            query, context_docs, mode, conversation_history
        )
        
        try:
            if stream:
                # Stream response
                response_text = ""
                stream_response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    options={
                        "temperature": config.OLLAMA_TEMPERATURE,
                        "num_predict": config.OLLAMA_MAX_TOKENS
                    }
                )
                
                for chunk in stream_response:
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        print(content, end="", flush=True)
                        response_text += content
                
                print()  # New line after streaming
                return response_text
            else:
                # Non-streaming response
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={
                        "temperature": config.OLLAMA_TEMPERATURE,
                        "num_predict": config.OLLAMA_MAX_TOKENS
                    }
                )
                
                return response["message"]["content"]
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"عذراً، حدث خطأ في توليد الإجابة. / Sorry, an error occurred generating the response.\n\nError: {str(e)}"
    
    def check_model_availability(self) -> bool:
        """
        Check if the configured model is available
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.client.list()
            model_names = [model["name"] for model in models.get("models", [])]
            return self.model in model_names
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False


if __name__ == "__main__":
    # Test Ollama connection
    client = OllamaClient()
    is_available = client.check_model_availability()
    print(f"Model {config.OLLAMA_MODEL} available: {is_available}")
