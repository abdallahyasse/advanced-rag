from app.memory.conversation_memory import ConversationMemory


class MemoryService:
    """
    Service for interacting with conversation memory.
    """

    def __init__(
        self,
        memory: ConversationMemory,
    ):
        self.memory = memory

    def remember_user(
        self,
        message: str,
    ):
        self.memory.add_user_message(message)

    def remember_assistant(
        self,
        message: str,
    ):
        self.memory.add_assistant_message(message)

    def history(self):
        return self.memory.get_history()

    def last_user_message(self):

        history = self.memory.get_history()

        for item in reversed(history):

            if item["role"] == "user":
                return item["content"]

        return None

    def last_assistant_message(self):

        history = self.memory.get_history()

        for item in reversed(history):

            if item["role"] == "assistant":
                return item["content"]

        return None

    def clear(self):
        self.memory.clear()