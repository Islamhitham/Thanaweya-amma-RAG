"""
Enhanced data processing with OCR and Structure-Aware Chunking
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Protocol
from PIL import Image
import io
import shutil
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config

# Try to import OCR capability
try:
    import pytesseract
    OCR_AVAILABLE = True
    if not shutil.which("tesseract"):
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default_path).exists():
            pytesseract.pytesseract.tesseract_cmd = default_path
            print(f"✓ Found Tesseract at: {default_path}")
        else:
            print("⚠ Tesseract not found in PATH or default location")
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not available.")

# --- CLEANING STRATEGIES ---

class TextCleaner(Protocol):
    def clean(self, text: str) -> str: pass

class BaseCleaner:
    def basic_clean(self, text: str) -> str:
        # Basic garbage removal before restructuring
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.isdigit() and len(s) < 4: continue # Page numbers
            if len(s) < 3 and not s in ['.', '!', '?']: continue # Noise
            if re.match(r'^[-_=]{3,}|^[—–]{3,}$', s): continue # Underscore/dash lines
            if re.match(r'^[-=_\s]+$', s): continue # Lines with just separators
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

class ArabicCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        text = self.basic_clean(text)
        # Remove diagram table artifacts
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            if line.count('|') > 2 or line.count('_') > 5: continue
            line = re.sub(r'\s[a-zA-Z]\s', ' ', line) # Isolated letters
            filtered_lines.append(line)
        text = '\n'.join(filtered_lines)
        text = re.sub(r'\s+([،؛؟])', r'\1', text)
        return text

class MathPhysicsCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        text = self.basic_clean(text)
        text = re.sub(r'^Fig\.?\s*\d+.*$', '', text, flags=re.MULTILINE)
        for op in ['=', '+', '-', '×', '÷']:
            text = text.replace(op, f" {op} ")
        text = re.sub(r'[ \t]+', ' ', text)
        return text

class ScienceCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        text = self.basic_clean(text)
        text = re.sub(r'^(Fig|Shape|Figure)\.?\s*\(?\d+\)?.*$', '', text, flags=re.MULTILINE|re.IGNORECASE)
        text = re.sub(r'^\s*[A-Z]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[·•●]\s*', '- ', text, flags=re.MULTILINE)
        return text

class EnglishCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        text = self.basic_clean(text)
        text = re.sub(r'\s+([A-D]\.)\s+', r'\n\1 ', text)
        return text

# --- STRUCTURE AWARE CHUNKING ---

class DocumentStructureChunker:
    """Chunks documents based on structural separators (Lessons, Chapters, etc.)"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Define structural separators for different languages/subjects
        self.separators = [
            # English
            r"^(Chapter|Unit|Lesson|Lecture)\s+\d+",
            r"^Section\s+\d+",
            # Arabic
            r"^(الباب|الفصل|الدرس|الوحدة|المحاضرة)\s+(الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|عشر|\d+)",
            r"^\d+\s*-\s*", # Numbered sections like "1 - Introduction"
        ]

    def _reconstruct_paragraphs(self, text: str) -> str:
        """Merge lines that shouldn't be broken (Paragraph Reconstruction)"""
        lines = text.split('\n')
        reconstructed = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    reconstructed.append(" ".join(current_para))
                    current_para = []
                continue
            
            # Check if this line looks like a header (Don't merge headers!)
            is_header = any(re.search(sep, line, re.IGNORECASE) for sep in self.separators)
            if is_header:
                if current_para:
                    reconstructed.append(" ".join(current_para))
                    current_para = []
                reconstructed.append(f"\n\n{line}\n\n") # Force separation
                continue

            current_para.append(line)
            
            # Heuristic: If line ends with period/colon, it might be end of para
            if line.endswith(('.', ':', '!', '?', '؟')):
                reconstructed.append(" ".join(current_para))
                current_para = []
                
        if current_para:
            reconstructed.append(" ".join(current_para))
            
        return "\n\n".join(reconstructed)

    def split_text(self, text: str) -> List[str]:
        # 1. Reconstruct broken paragraphs first
        clean_struct_text = self._reconstruct_paragraphs(text)
        
        # 2. Split by major sections first (using double newlines inserted by reconstructor)
        # Actually, RecursiveCharacterTextSplitter handles this if we give it the right separators
        # But we want to ensure headers start new chunks if possible.
        
        return self.text_splitter.split_text(clean_struct_text)

# --- PIPELINE ---

class DocumentProcessor:
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.chunker = DocumentStructureChunker()
        self.cleaners = {
            'arabic': ArabicCleaner(),
            'math': MathPhysicsCleaner(),
            'physics': MathPhysicsCleaner(),
            'chemistry': ScienceCleaner(),
            'biology': ScienceCleaner(),
            'english': EnglishCleaner()
        }
        
    def get_cleaner(self, subject: str) -> TextCleaner:
        return self.cleaners.get(subject.lower(), EnglishCleaner())
    
    def extract_text_with_ocr(self, page: fitz.Page, lang: str = 'ara+eng') -> str:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            return pytesseract.image_to_string(img, lang=lang)
        except Exception: return ""

    def process_pdf(self, pdf_path: Path, subject: str) -> str:
        doc = fitz.open(pdf_path)
        full_text = ""
        print(f"    - Processing {pdf_path.name} ({len(doc)} pages)")
        for i, page in enumerate(doc):
            text = page.get_text()
            needs_ocr = len(text.strip()) < 50 or subject == 'arabic'
            if needs_ocr and self.use_ocr:
                lang = 'ara+eng' if subject == 'arabic' else 'eng'
                text = self.extract_text_with_ocr(page, lang)
            full_text += f"\n{text}"
            if (i+1) % 20 == 0: print(f"      Running OCR on page {i+1}...")
        doc.close()
        cleaner = self.get_cleaner(subject)
        return cleaner.clean(full_text)

    def process_element_chunks(self, text: str, pdf_file: Path, subject: str) -> List[Dict]:
        """Process text into chunks"""
        chunks = self.chunker.split_text(text)
        doc_chunks = []
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50: continue
            doc_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "subject": subject, 
                    "filename": pdf_file.name, 
                    "source": str(pdf_file),
                    "chunk_id": idx
                }
            })
        return doc_chunks

    def process_subject_pdfs(self, subject: str) -> List[Dict]:
        subject_dir = config.DATA_DIR / subject
        if not subject_dir.exists(): return []
        all_chunks = []
        for pdf_file in subject_dir.glob("*.pdf"):
            try:
                text = self.process_pdf(pdf_file, subject)
                print("      Chunking by structure...")
                doc_chunks = self.process_element_chunks(text, pdf_file, subject)
                all_chunks.extend(doc_chunks)
                print(f"    ✓ Created {len(doc_chunks)} structure-aware chunks")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback; traceback.print_exc()
        return all_chunks

    def process_all_subjects(self) -> Dict[str, List[Dict]]:
        all_data = {}
        for subject in config.SUBJECTS:
            print(f"\nSubject: {subject.upper()}")
            all_data[subject] = self.process_subject_pdfs(subject)
        return all_data

if __name__ == "__main__":
    processor = DocumentProcessor()
