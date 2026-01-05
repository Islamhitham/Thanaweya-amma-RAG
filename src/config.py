"""
Configuration file for Thanaweya Amma RAG System
"""
import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from src/ to project root
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Subjects
SUBJECTS = ["arabic", "math", "chemistry", "biology", "english", "physics"]

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "thanaweya_amma_docs"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Supports Arabic
EMBEDDING_DIMENSION = 384

# Document Chunking
CHUNK_SIZE = 600  # Characters
CHUNK_OVERLAP = 200  # Characters overlap between chunks

# Ollama Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 2000

# Retrieval Configuration
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
HYBRID_SEARCH_WEIGHTS = {
    "semantic": 0.6,  # Weight for semantic/embedding search
    "bm25": 0.4       # Weight for BM25 lexical search
}

# Conversation Memory
MAX_MEMORY_MESSAGES = 3  # Last 3 messages (user + assistant pairs)

# System Prompts
SYSTEM_PROMPT_QA = """أنت مساعد تعليمي متخصص في مساعدة طلاب الثانوية العامة المصرية.
استخدم المعلومات المقدمة للإجابة على أسئلة الطالب بدقة ووضوح.
إذا لم تكن المعلومات كافية، قل ذلك بوضوح.

You are an educational assistant specialized in helping Egyptian Thanaweya Amma students.
Use the provided information to answer student questions accurately and clearly.
If the information is not sufficient, say so clearly."""

SYSTEM_PROMPT_QUIZ = """أنت مساعد تعليمي متخصص في إنشاء اختبارات لطلاب الثانوية العامة المصرية.
قم بإنشاء أسئلة اختيار من متعدد بناءً على المحتوى المقدم.
تأكد من أن الأسئلة واضحة والخيارات معقولة.

You are an educational assistant specialized in creating quizzes for Egyptian Thanaweya Amma students.
Create multiple choice questions based on the provided content.
Ensure questions are clear and options are reasonable."""

SYSTEM_PROMPT_EXPLAIN = """أنت مساعد تعليمي متخصص في شرح المفاهيم لطلاب الثانوية العامة المصرية.
اشرح الموضوع المطلوب بطريقة واضحة ومبسطة مع أمثلة عند الحاجة.

You are an educational assistant specialized in explaining concepts to Egyptian Thanaweya Amma students.
Explain the requested topic clearly and simply with examples when needed."""

# Logging
LOG_LEVEL = "INFO"
