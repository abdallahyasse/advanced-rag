from app.agentic_rag.reflector import ReflectionAgent
from app.agentic_rag.query_rewriter import QueryRewriter
from app.container.dependencies import container


reflector = ReflectionAgent()
rewriter = QueryRewriter()


def reflection_node(state):

    result = reflector.reflect(
        state["question"]
    )

    state["needs_rewrite"] = result.needs_rewrite

    return state


def rewrite_node(state):

    rewritten = rewriter.rewrite(
        state["question"]
    )

    state["rewritten_question"] = (
        rewritten.rewritten_question
    )

    return state


def rag_node(state):

    question = state.get(
        "rewritten_question"
    ) or state["question"]

    answer = container.rag_service.ask(
        question=question,
        top_k=3,
    )

    state["answer"] = answer

    return state