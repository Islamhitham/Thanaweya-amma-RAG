"""
Main application for Thanaweya Amma RAG System
"""
import sys
from pathlib import Path
from colorama import Fore, Style, init

from src.config import *
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever
from src.llm_client import OllamaClient
from src.conversation_memory import ConversationMemory

# Initialize colorama for colored terminal output
init(autoreset=True)


class ThanaweayaRAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        """Initialize all components"""
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Thanaweya Amma RAG System - Initializing...")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        # Initialize vector store
        print(f"{Fore.YELLOW}[1/3] Initializing vector store...")
        self.vector_store = VectorStore()
        stats = self.vector_store.get_collection_stats()
        print(f"{Fore.GREEN}✓ Vector store ready: {stats['total_documents']} documents indexed\n")
        
        # Initialize hybrid retriever
        print(f"{Fore.YELLOW}[2/3] Initializing hybrid retriever...")
        self.retriever = HybridRetriever(self.vector_store)
        print(f"{Fore.GREEN}✓ Hybrid retriever ready\n")
        
        # Initialize LLM client
        print(f"{Fore.YELLOW}[3/3] Connecting to Ollama...")
        self.llm_client = OllamaClient()
        print(f"{Fore.GREEN}✓ Ollama client ready\n")
        
        # Conversation memory
        self.memory = ConversationMemory()
        
        # Current session state
        self.current_subject = None
        self.current_mode = None
        
        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}System ready!")
        print(f"{Fore.GREEN}{'='*60}\n")
    
    def select_subject(self) -> str:
        """
        Interactive subject selection
        
        Returns:
            Selected subject name
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Available Subjects:")
        print(f"{Fore.CYAN}{'='*60}")
        
        for i, subject in enumerate(config.SUBJECTS, 1):
            print(f"{Fore.YELLOW}{i}. {subject.upper()}")
        
        while True:
            try:
                choice = input(f"\n{Fore.WHITE}Select subject (1-{len(config.SUBJECTS)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print(f"{Fore.RED}Exiting...")
                    sys.exit(0)
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(config.SUBJECTS):
                    subject = config.SUBJECTS[choice_idx]
                    print(f"{Fore.GREEN}✓ Selected: {subject.upper()}\n")
                    return subject
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")
            except ValueError:
                print(f"{Fore.RED}Please enter a number.")
    
    def select_mode(self) -> str:
        """
        Interactive mode selection
        
        Returns:
            Selected mode ('ask', 'quiz', or 'explain')
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Select Mode:")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}1. ASK - Ask questions and get answers")
        print(f"{Fore.YELLOW}2. QUIZ - Generate practice quizzes")
        print(f"{Fore.YELLOW}3. EXPLAIN - Get detailed explanations")
        
        modes = {"1": "ask", "2": "quiz", "3": "explain"}
        
        while True:
            choice = input(f"\n{Fore.WHITE}Select mode (1-3): ").strip()
            
            if choice in modes:
                mode = modes[choice]
                mode_names = {"ask": "Q&A", "quiz": "Quiz Generation", "explain": "Explanation"}
                print(f"{Fore.GREEN}✓ Mode: {mode_names[mode]}\n")
                return mode
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 1, 2, or 3.")
    
    def chat_loop(self):
        """Main chat loop with memory"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Chat Session Started")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}Subject: {self.current_subject.upper()}")
        print(f"{Fore.YELLOW}Mode: {self.current_mode.upper()}")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}Type your message or commands:")
        print(f"{Fore.WHITE}  - 'clear' to clear conversation memory")
        print(f"{Fore.WHITE}  - 'change' to change subject/mode")
        print(f"{Fore.WHITE}  - 'quit' to exit")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print(f"\n{Fore.YELLOW}Goodbye! Happy studying! 📚")
                sys.exit(0)
            
            if user_input.lower() == 'clear':
                self.memory.clear()
                print(f"{Fore.YELLOW}✓ Conversation memory cleared.\n")
                continue
            
            if user_input.lower() == 'change':
                self.current_subject = self.select_subject()
                self.current_mode = self.select_mode()
                self.memory.clear()
                print(f"{Fore.YELLOW}✓ Subject and mode changed. Memory cleared.\n")
                continue
            
            # Process query
            print(f"\n{Fore.CYAN}[Retrieving relevant content...]{Style.RESET_ALL}")
            
            # Retrieve relevant documents
            context_docs = self.retriever.search(
                query=user_input,
                subject=self.current_subject,
                top_k=config.TOP_K_RETRIEVAL
            )
            
            if not context_docs:
                print(f"{Fore.RED}⚠ No relevant content found for your query.\n")
                continue
            
            print(f"{Fore.GREEN}✓ Found {len(context_docs)} relevant sources\n")
            
            # Get conversation history
            conversation_history = self.memory.get_formatted_history()
            
            # Generate response
            print(f"{Fore.CYAN}Assistant: {Style.RESET_ALL}", end="")
            response = self.llm_client.generate_response(
                query=user_input,
                context_docs=context_docs,
                mode=self.current_mode,
                conversation_history=conversation_history,
                stream=True
            )
            
            # Add to memory
            self.memory.add_interaction(user_input, response)
            
            print(f"\n{Fore.MAGENTA}[Memory: {self.memory.get_message_count()}/{config.MAX_MEMORY_MESSAGES} conversations]{Style.RESET_ALL}\n")
    
    def run(self):
        """Run the main application"""
        try:
            # Welcome message
            print(f"\n{Fore.CYAN}╔{'═'*58}╗")
            print(f"{Fore.CYAN}║{' '*58}║")
            print(f"{Fore.CYAN}║{' '*10}Welcome to Thanaweya Amma RAG System{' '*10}║")
            print(f"{Fore.CYAN}║{' '*58}║")
            print(f"{Fore.CYAN}╚{'═'*58}╝\n")
            
            # Select subject and mode
            self.current_subject = self.select_subject()
            self.current_mode = self.select_mode()
            
            # Start chat
            self.chat_loop()
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Session interrupted. Goodbye! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}")
            sys.exit(1)


def main():
    """Entry point"""
    # Check if ChromaDB has data
    if not config.CHROMA_DIR.exists():
        print(f"{Fore.RED}{'='*60}")
        print(f"{Fore.RED}ERROR: ChromaDB not found!")
        print(f"{Fore.RED}{'='*60}")
        print(f"{Fore.YELLOW}Please run 'python ingest_data.py' first to load your PDFs.")
        print(f"{Fore.YELLOW}Make sure your PDFs are in the 'data/<subject>/' directories.\n")
        sys.exit(1)
    
    # Run the system
    system = ThanaweayaRAGSystem()
    system.run()


if __name__ == "__main__":
    main()
