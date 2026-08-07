from app.memory.memory_service import MemoryService


class ContextBuilder:
    """
    Builds conversational context for the Agent.
    """

    def __init__(
        self,
        memory_service: MemoryService,
    ):
        self.memory = memory_service

    def build_context(
        self,
        current_question: str,
        max_turns: int = 4,
    ) -> str:

        history = self.memory.history()

        if not history:
            return current_question

        recent = history[-max_turns:]

        context = []

        for message in recent:

            role = message["role"].capitalize()

            context.append(
                f"{role}: {message['content']}"
            )

        context.append(
            f"User: {current_question}"
        )

        return "\n".join(context)