"""
Conversation memory to maintain last N messages
"""
from typing import List, Dict
from collections import deque
from . import config


class ConversationMemory:
    """Manages conversation history with a sliding window"""
    
    def __init__(self, max_messages: int = config.MAX_MEMORY_MESSAGES):
        """
        Initialize conversation memory
        
        Args:
            max_messages: Maximum number of message pairs to remember
        """
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)
    
    def add_interaction(self, user_message: str, assistant_message: str):
        """
        Add a user-assistant interaction to memory
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
        """
        self.messages.append({
            "user": user_message,
            "assistant": assistant_message
        })
    
    def get_formatted_history(self) -> str:
        """
        Get formatted conversation history
        
        Returns:
            Formatted string of conversation history
        """
        if not self.messages:
            return ""
        
        history_parts = []
        for i, interaction in enumerate(self.messages, 1):
            history_parts.append(f"User: {interaction['user']}")
            history_parts.append(f"Assistant: {interaction['assistant']}")
        
        return "\n".join(history_parts)
    
    def get_messages(self) -> List[Dict]:
        """
        Get raw message list
        
        Returns:
            List of message dictionaries
        """
        return list(self.messages)
    
    def clear(self):
        """Clear all conversation history"""
        self.messages.clear()
    
    def is_empty(self) -> bool:
        """Check if memory is empty"""
        return len(self.messages) == 0
    
    def get_message_count(self) -> int:
        """Get current number of stored messages"""
        return len(self.messages)


if __name__ == "__main__":
    # Test conversation memory
    memory = ConversationMemory(max_messages=3)
    
    memory.add_interaction("What is photosynthesis?", "Photosynthesis is...")
    memory.add_interaction("Give me an example", "For example...")
    memory.add_interaction("Thanks!", "You're welcome!")
    memory.add_interaction("Another question?", "Sure, ask away!")  # This should push out the first
    
    print(f"Message count: {memory.get_message_count()}")
    print("\nFormatted history:")
    print(memory.get_formatted_history())
