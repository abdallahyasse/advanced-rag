from app.container.dependencies import container

from app.agentic_rag.reflector import ReflectionAgent
from app.agentic_rag.query_rewriter import QueryRewriter
from app.agentic_rag.evaluator import EvaluatorAgent
from app.agentic_rag.search_rewriter import SearchQueryRewriter

from app.agentic_rag.retry_controller import RetryState
from app.agentic_rag.retrieval_guard import RetrievalGuard


class AgenticRAGWorkflow:
    """
    Production Agentic RAG Workflow

    User
        ↓
    Reflection
        ↓
    Rewrite
        ↓
    Retrieve
        ↓
    Generate
        ↓
    Evaluate
        ↓
    Retry (if needed)
        ↓
    Save Memory
    """

    def __init__(self):

        self.reflector = ReflectionAgent()

        self.rewriter = QueryRewriter()

        self.search_rewriter = SearchQueryRewriter()

        self.evaluator = EvaluatorAgent()

    def run(
        self,
        question: str,
    ):

        # -----------------------------
        # Save User Message
        # -----------------------------
        container.memory_service.remember_user(
            question
        )

        # -----------------------------
        # Reflection
        # -----------------------------
        reflection = self.reflector.reflect(
            question
        )

        print("\n========== Reflection ==========")
        print(reflection)

        rewritten_question = question

        # -----------------------------
        # Rewrite
        # -----------------------------
        if reflection.needs_rewrite:

            rewrite = self.rewriter.rewrite(
                question
            )

            rewritten_question = (
                rewrite.rewritten_question
            )

            print("\n========== Rewritten ==========")
            print(rewrite)

        # -----------------------------
        # Initial Retrieval
        # -----------------------------
        answer = container.rag_service.ask(
            rewritten_question
        )

        # -----------------------------
        # Initial Evaluation
        # -----------------------------
        evaluation = self.evaluator.evaluate(
            question=rewritten_question,
            context=answer,
        )

        print("\n========== Evaluation ==========")
        print(evaluation)

        # -----------------------------
        # Retry Controller
        # -----------------------------
        retry_state = RetryState()

        previous_hash = RetrievalGuard.fingerprint(
            answer
        )

        while not evaluation.enough_context:

            if not retry_state.can_retry():

                print(
                    "\n========== Max Retry =========="
                )

                break

            retry_state.increase()

            print(
                f"\n========== Retry {retry_state.retries} =========="
            )

            retry = self.search_rewriter.rewrite(
                question=rewritten_question,
                reason=evaluation.reason,
            )

            rewritten_question = (
                retry.rewritten_question
            )

            print(retry)

            answer = container.rag_service.ask(
                rewritten_question
            )

            current_hash = RetrievalGuard.fingerprint(
                answer
            )

            if current_hash == previous_hash:

                print(
                    "\n========== Same Retrieval =========="
                )

                break

            previous_hash = current_hash

            evaluation = self.evaluator.evaluate(
                question=rewritten_question,
                context=answer,
            )

            print(
                "\n========== Evaluation =========="
            )

            print(evaluation)

        # -----------------------------
        # Save Assistant Message
        # -----------------------------
        container.memory_service.remember_assistant(
            answer
        )

        # -----------------------------
        # Final Response
        # -----------------------------
        return {
            "question": question,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "evaluation": evaluation,
            "retry_count": retry_state.retries,
        }