#  Thanaweya Amma AI Tutor (RAG System)

A powerful, local Retrieval-Augmented Generation (RAG) system specialized for the **Egyptian High School (Thanaweya Amma) Curriculum**. 

This project ingests PDF textbooks (Arabic & English), processes them with a **smart hybrid pipeline**, and uses **Ollama** to provide accurate, context-aware answers to student questions.

##  Key Features

###  Intelligent Data Processing
*   **Hybrid Extraction Pipeline**: Automatically switches between **Text Layer extraction** (for high-quality vector PDFs) and **Tesseract OCR** (for scanned/image-based files) to ensure zero data loss.
*   **Structure-Aware Chunking**: 
    *   Does NOT just split by character count.
    *   **Reconstructs Paragraphs**: Merges broken generic lines from OCR into coherent sentences.
    *   **Respects Hierarchy**: Splits documents by logical units (Chapters, Lessons, Units, Abwab).
*   **Subject-Specific Cleaning**:
    *   **Arabic**: Removes diagram noise, fixes punctuation, and handles mixed-language text.
    *   **Math/Physics**: Preserves equations, symbols (`+`, `=`, `×`), and operator spacing.
    *   **Biology/Chemistry**: Handles scientific notation and removes figure captions.
*   **Column Detection**: Automatically handles complex 2-column layouts in textbooks.

### Advanced Retrieval System
*   **Hybrid Search**: Combines two powerful methods using **Reciprocal Rank Fusion (RRF)**:
    1.  **Semantic Search (Dense)**: To understand *meaning* (e.g., "Photosynthesis" matches "Process of making food").
    2.  **Lexical Search (Sparse)**: Using **BM25** to match specific *keywords* and scientific terms.

### System Modes & Prompts

The system supports three distinct interaction modes, each with a specialized bilingual prompt:

### 1. Q&A Mode (Ask)
### 2. Quiz Mode (Generate Quiz)
### 3. Explain Mode (Simplify)

### Local AI Engine
*   **Privacy-First**: Runs entirely locally using **Ollama** and **ChromaDB**. No data leaves your machine.
*   **Contextual Memory**: Remembers the last 3 turns of conversation for follow-up questions.

## Data Source
The curriculum data used to build this system is sourced from the official **Egyptian Ministry of Education E-Library**:
 **[https://ellibrary.moe.gov.eg/books/](https://ellibrary.moe.gov.eg/books/)**

##  Installation

### Prerequisites
1.  **Python 3.11** (Conda recommended)
2.  **Ollama**: [Download Here](https://ollama.com/)
3.  **Tesseract OCR**: [Installation Guide](https://github.com/tesseract-ocr/tessdoc)
    *   *Windows*: Install `tesseract-ocr-w64-setup.exe` to `C:\Program Files\Tesseract-OCR`.

### Setup
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/thanaweya-amma-rag.git
    cd thanaweya-amma-rag
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download AI Models**
    ```bash
    ollama pull llama3.1:8b
    # The embedding model will download automatically on first run
    ```

## Usage

### 1. Ingest Data
Put your PDFs in `data/{subject}` folders (e.g., `data/physics/book.pdf`).
Then run the ingestion pipeline:
```bash
python ingest_data.py
```
*This will extract, clean, and chunk all documents into the vector database.*

### 2. Run the AI Tutor
Start the interactive chat interface:
```bash
python main.py
```
*   Select a subject (e.g., Physics).
*   Ask questions like: "Explain Ohm's Law" or "What are the rules of Arabic grammar?"

##  Project Structure
*   `src/data_processing.py`: Core logic for OCR, cleaning, and structural chunking.
*   `src/hybrid_retriever.py`: Implements the RRF algorithm combining BM25 + ChromaDB.
*   `src/llm_client.py`: Interface for Ollama generation.
*   `scripts/`: Utility scripts for debugging chunks and testing retrieval.

##  Contributing
Open to contributions! Please ensure any PRs maintaining the high quality of the "Structure-Aware Chunker" logic.
