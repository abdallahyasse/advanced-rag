from app.container.dependencies import container

from app.agentic_rag.reflector import ReflectionAgent
from app.agentic_rag.query_rewriter import QueryRewriter
from app.agentic_rag.evaluator import EvaluatorAgent
from app.agentic_rag.search_rewriter import SearchQueryRewriter

from app.agentic_rag.retry_controller import RetryState
from app.agentic_rag.retrieval_guard import RetrievalGuard


class AgenticRAGWorkflow:
    """
    Agentic RAG workflow.

    Flow:
        Original Question
            -> Reflection
            -> Optional Query Rewrite
            -> Retrieval
            -> Evidence Evaluation
            -> Bounded Search Retries
            -> Safe Final Answer
            -> Save Conversation Memory
    """

    def __init__(self):
        self.reflector = ReflectionAgent()
        self.rewriter = QueryRewriter()
        self.search_rewriter = SearchQueryRewriter()
        self.evaluator = EvaluatorAgent()

    def run(self, question: str) -> dict:
        original_question = question
        search_question = question

        # 1. Save the user question
        container.memory_service.remember_user(original_question)

        # 2. Reflection
        reflection = self.reflector.reflect(original_question)

        print("\n========== Reflection ==========")
        print(reflection)

        # 3. Optional conversational query rewrite
        if reflection.needs_rewrite:
            conversation_context = (
                container.context_builder.build_context(
                    original_question
                )
            )

            rewrite = self.rewriter.rewrite(
                question=original_question,
                conversation_context=conversation_context,
            )

            search_question = rewrite.rewritten_question

            print("\n========== Query Rewrite ==========")
            print(rewrite)

        # 4. Initial retrieval
        top_k = 3

        rag_result = container.rag_service.ask(
            search_question,
            top_k=top_k,
        )

        # Always evaluate evidence against the original question.
        evaluation = self.evaluator.evaluate(
            question=original_question,
            context=rag_result["context"],
        )

        print("\n========== Initial Evaluation ==========")
        print(evaluation)

        retry_state = RetryState()
        previous_hash = RetrievalGuard.fingerprint(
            rag_result["context"]
        )

        # 5. Bounded retry loop
        while not evaluation.enough_context and retry_state.can_retry():
            retry_state.increase()

            print(
                f"\n========== Retry {retry_state.retries} =========="
            )

            retry = self.search_rewriter.rewrite(
                question=search_question,
                reason=evaluation.reason,
            )

            candidate_question = retry.rewritten_question

            print("\n========== Search Query Rewrite ==========")
            print(retry)

            # Increase retrieval breadth gradually.
            top_k = min(top_k + 2, 12)

            print(
                f"\n========== Retrieval: top_k={top_k} =========="
            )

            candidate_result = container.rag_service.ask(
                candidate_question,
                top_k=top_k,
            )

            current_hash = RetrievalGuard.fingerprint(
                candidate_result["context"]
            )

            # Stop if this retry adds no new context.
            if current_hash == previous_hash:
                print("\n========== Same Context: Stop Retry ==========")
                break

            # Accept the new retrieval result.
            search_question = candidate_question
            rag_result = candidate_result
            previous_hash = current_hash

            # Evaluate against the original user question,
            # not the rewritten search query.
            evaluation = self.evaluator.evaluate(
                question=original_question,
                context=rag_result["context"],
            )

            print("\n========== Re-Evaluation ==========")
            print(evaluation)

        # 6. Final answer guard
        if not evaluation.enough_context:
            answer = (
                "I couldn't find enough information in the available "
                "documents to answer this question reliably."
            )

            print("\n========== Final Answer Guard ==========")
            print("Insufficient evidence. Using safe fallback.")
        else:
            answer = rag_result["answer"]

        # 7. Save assistant response
        container.memory_service.remember_assistant(answer)

        # 8. Return response and diagnostics
        return {
            "question": original_question,
            "rewritten_question": search_question,
            "answer": answer,
            "sources": rag_result["sources"],
            "documents": rag_result["documents"],
            "evaluation": evaluation,
            "retry_count": retry_state.retries,
        }