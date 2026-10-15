
from app.agentic_rag.evaluator import EvaluatorAgent
from app.agentic_rag.models import EvaluationResult
from app.agentic_rag.retrieval_guard import RetrievalGuard
from app.agentic_rag.retry_controller import RetryState


def test_retry_state_allows_only_max_retries():
    state = RetryState(max_retries=2)

    assert state.can_retry() is True

    state.increase()
    assert state.retries == 1
    assert state.can_retry() is True

    state.increase()
    assert state.retries == 2
    assert state.can_retry() is False


def test_retry_state_with_zero_max_retries():
    state = RetryState(max_retries=0)

    assert state.can_retry() is False


def test_retrieval_guard_same_context_has_same_fingerprint():
    context = "Python is a programming language."

    first = RetrievalGuard.fingerprint(context)
    second = RetrievalGuard.fingerprint(context)

    assert first == second


def test_retrieval_guard_different_context_has_different_fingerprint():
    first = RetrievalGuard.fingerprint("Python is a language.")
    second = RetrievalGuard.fingerprint("Java is a language.")

    assert first != second


def test_evaluator_rejects_empty_context_without_calling_llm(monkeypatch):
    evaluator = EvaluatorAgent()

    def unexpected_chat(*args, **kwargs):
        raise AssertionError(
            "LLM should not be called when context is empty."
        )

    monkeypatch.setattr(evaluator.llm, "chat", unexpected_chat)

    result = evaluator.evaluate(
        question="What is Python?",
        context="   ",
    )

    assert result == EvaluationResult(
        enough_context=False,
        reason="The retrieved context is empty.",
    )

import importlib
import sys
from types import ModuleType, SimpleNamespace

from app.agentic_rag.models import EvaluationResult, ReflectionResult


def make_fake_container(monkeypatch):
    memory_service = SimpleNamespace(
        remember_user=lambda question: None,
        remember_assistant=lambda answer: None,
    )

    fake_container = SimpleNamespace(
        memory_service=memory_service,
        context_builder=SimpleNamespace(
            build_context=lambda question: ""
        ),
        rag_service=None,
    )

    # Prevent importing dependencies.py from loading real PDFs and indexes.
    fake_dependencies = ModuleType("app.container.dependencies")
    fake_dependencies.container = fake_container

    monkeypatch.setitem(
        sys.modules,
        "app.container.dependencies",
        fake_dependencies,
    )

    workflow_module = importlib.import_module(
        "app.agentic_rag.workflow"
    )

    monkeypatch.setattr(
        workflow_module,
        "container",
        fake_container,
    )

    return workflow_module, fake_container


def test_workflow_returns_answer_when_initial_context_is_sufficient(
    monkeypatch,
):
    workflow_module, fake_container = make_fake_container(monkeypatch)

    fake_container.rag_service = SimpleNamespace(
        ask=lambda question, top_k: {
            "answer": "Python is a programming language.",
            "context": "Python is a programming language.",
            "sources": [{"source": "test.pdf", "page": 1, "score": 0.9}],
            "documents": [{"text": "Python is a programming language."}],
        }
    )

    monkeypatch.setattr(
        workflow_module.ReflectionAgent,
        "reflect",
        lambda self, question: ReflectionResult(
            clear=True,
            needs_rewrite=False,
            reason="The question is clear.",
        ),
    )

    monkeypatch.setattr(
        workflow_module.EvaluatorAgent,
        "evaluate",
        lambda self, question, context: EvaluationResult(
            enough_context=True,
            reason="The context answers the question.",
        ),
    )

    workflow = workflow_module.AgenticRAGWorkflow()
    result = workflow.run("What is Python?")

    assert result["answer"] == "Python is a programming language."
    assert result["retry_count"] == 0
    assert result["evaluation"].enough_context is True
    assert result["sources"] == [
        {"source": "test.pdf", "page": 1, "score": 0.9}
    ]


def test_workflow_returns_safe_fallback_when_context_is_insufficient(
    monkeypatch,
):
    workflow_module, fake_container = make_fake_container(monkeypatch)

    fake_container.rag_service = SimpleNamespace(
        ask=lambda question, top_k: {
            "answer": "Possibly an unsupported answer.",
            "context": "Unrelated context.",
            "sources": [],
            "documents": [],
        }
    )

    monkeypatch.setattr(
        workflow_module.ReflectionAgent,
        "reflect",
        lambda self, question: ReflectionResult(
            clear=True,
            needs_rewrite=False,
            reason="The question is clear.",
        ),
    )

    monkeypatch.setattr(
        workflow_module.EvaluatorAgent,
        "evaluate",
        lambda self, question, context: EvaluationResult(
            enough_context=False,
            reason="The context does not contain the answer.",
        ),
    )

    # Avoid retries in this test so it focuses on the final safety guard.
    monkeypatch.setattr(
        workflow_module.RetryState,
        "can_retry",
        lambda self: False,
    )

    workflow = workflow_module.AgenticRAGWorkflow()
    result = workflow.run("What is Python?")

    assert result["answer"] == (
        "I couldn't find enough information in the available "
        "documents to answer this question reliably."
    )
    assert result["retry_count"] == 0
    assert result["evaluation"].enough_context is False

def test_workflow_retry_succeeds_when_new_context_is_sufficient(monkeypatch):
    workflow_module, fake_container = make_fake_container(monkeypatch)

    responses = [
        {
            "answer": "Unsupported answer.",
            "context": "Unrelated information.",
            "sources": [],
            "documents": [],
        },
        {
            "answer": "Python is a programming language.",
            "context": "Python is a programming language.",
            "sources": [{"source": "test.pdf", "page": 1, "score": 0.9}],
            "documents": [{"text": "Python is a programming language."}],
        },
    ]
    calls = []

    def fake_ask(question, top_k):
        calls.append({"question": question, "top_k": top_k})
        return responses[len(calls) - 1]

    fake_container.rag_service = SimpleNamespace(ask=fake_ask)

    monkeypatch.setattr(
        workflow_module.ReflectionAgent,
        "reflect",
        lambda self, question: ReflectionResult(
            clear=True,
            needs_rewrite=False,
            reason="The question is clear.",
        ),
    )
    monkeypatch.setattr(
        workflow_module.SearchQueryRewriter,
        "rewrite",
        lambda self, question, reason: SimpleNamespace(
            rewritten_question="What is Python?"
        ),
    )

    evaluations = iter([
        EvaluationResult(
            enough_context=False,
            reason="The initial context is insufficient.",
        ),
        EvaluationResult(
            enough_context=True,
            reason="The new context answers the question.",
        ),
    ])
    evaluated_questions = []

    def fake_evaluate(self, question, context):
        evaluated_questions.append(question)
        return next(evaluations)

    monkeypatch.setattr(
        workflow_module.EvaluatorAgent,
        "evaluate",
        fake_evaluate,
    )

    workflow = workflow_module.AgenticRAGWorkflow()
    result = workflow.run("What is Python?")

    assert result["answer"] == "Python is a programming language."
    assert result["retry_count"] == 1
    assert result["evaluation"].enough_context is True
    assert [call["top_k"] for call in calls] == [3, 5]
    assert evaluated_questions == ["What is Python?", "What is Python?"]


def test_workflow_stops_when_retry_returns_same_context(monkeypatch):
    workflow_module, fake_container = make_fake_container(monkeypatch)

    context = "Unrelated information."
    calls = []

    def fake_ask(question, top_k):
        calls.append({"question": question, "top_k": top_k})
        return {
            "answer": "Unsupported answer.",
            "context": context,
            "sources": [],
            "documents": [],
        }

    fake_container.rag_service = SimpleNamespace(ask=fake_ask)

    monkeypatch.setattr(
        workflow_module.ReflectionAgent,
        "reflect",
        lambda self, question: ReflectionResult(
            clear=True,
            needs_rewrite=False,
            reason="The question is clear.",
        ),
    )
    monkeypatch.setattr(
        workflow_module.SearchQueryRewriter,
        "rewrite",
        lambda self, question, reason: SimpleNamespace(
            rewritten_question="Alternative search query"
        ),
    )

    evaluation_calls = []

    def fake_evaluate(self, question, context):
        evaluation_calls.append(question)
        return EvaluationResult(
            enough_context=False,
            reason="The context does not answer the question.",
        )

    monkeypatch.setattr(
        workflow_module.EvaluatorAgent,
        "evaluate",
        fake_evaluate,
    )

    workflow = workflow_module.AgenticRAGWorkflow()
    result = workflow.run("What is Python?")

    assert result["retry_count"] == 1
    assert len(calls) == 2
    assert len(evaluation_calls) == 1
    assert result["answer"] == (
        "I couldn't find enough information in the available "
        "documents to answer this question reliably."
    )


def test_workflow_respects_maximum_retry_count(monkeypatch):
    workflow_module, fake_container = make_fake_container(monkeypatch)

    calls = []

    def fake_ask(question, top_k):
        calls.append({"question": question, "top_k": top_k})
        return {
            "answer": "Unsupported answer.",
            "context": f"Different context number {len(calls)}",
            "sources": [],
            "documents": [],
        }

    fake_container.rag_service = SimpleNamespace(ask=fake_ask)

    monkeypatch.setattr(
        workflow_module.ReflectionAgent,
        "reflect",
        lambda self, question: ReflectionResult(
            clear=True,
            needs_rewrite=False,
            reason="The question is clear.",
        ),
    )
    monkeypatch.setattr(
        workflow_module.SearchQueryRewriter,
        "rewrite",
        lambda self, question, reason: SimpleNamespace(
            rewritten_question=f"Search attempt {len(calls) + 1}"
        ),
    )
    monkeypatch.setattr(
        workflow_module.EvaluatorAgent,
        "evaluate",
        lambda self, question, context: EvaluationResult(
            enough_context=False,
            reason="The context is insufficient.",
        ),
    )

    workflow = workflow_module.AgenticRAGWorkflow()
    result = workflow.run("What is Python?")

    assert result["retry_count"] == 2
    assert len(calls) == 3
    assert [call["top_k"] for call in calls] == [3, 5, 7]
    assert result["answer"] == (
        "I couldn't find enough information in the available "
        "documents to answer this question reliably."
    )
