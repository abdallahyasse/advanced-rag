from collections import deque


class ConversationMemory:
    """
    Simple in-memory conversation history.
    """

    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)

    def add_user_message(self, message: str):
        self.history.append(
            {
                "role": "user",
                "content": message,
            }
        )

    def add_assistant_message(self, message: str):
        self.history.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

    def get_history(self):
        return list(self.history)

    def clear(self):
        self.history.clear()