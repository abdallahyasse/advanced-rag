
from unittest.mock import Mock

import pytest

from app.agent.agent import Agent
from app.agent.executor import ToolExecutor
from app.agent.models import FinalResponse, ToolResult
from app.agent.planner import Planner
from app.agent.tools.registry import TOOLS
from app.llm.models import ToolResponse


def test_planner_falls_back_to_rag_for_invalid_tool():
    planner = Planner()
    planner.llm = Mock()

    planner.llm.chat.return_value = ToolResponse(
        thought="Invalid tool selected.",
        tool="refusal",
        arguments={},
    )

    result = planner.plan("What is an unknown fact?")

    assert result.tool == "rag"
    assert result.arguments == {
        "question": "What is an unknown fact?"
    }


def test_executor_rejects_unknown_tool():
    result = ToolExecutor().execute(
        tool_name="unknown_tool",
        arguments={},
    )

    assert result.success is False
    assert result.tool == "unknown_tool"
    assert "Unknown tool" in result.error


def test_executor_handles_invalid_arguments():
    result = ToolExecutor().execute(
        tool_name="calculator",
        arguments={},
    )

    assert result.success is False
    assert result.tool == "calculator"
    assert "Invalid tool arguments" in result.error


def test_executor_returns_tool_failure():
    class FailingTool:
        def run(self, **kwargs):
            return ToolResult(
                success=False,
                tool="test",
                error="Simulated failure",
            )

    executor = ToolExecutor()

    original_tool = TOOLS.get("test_tool")
    TOOLS["test_tool"] = FailingTool()

    try:
        result = executor.execute("test_tool", {})
    finally:
        if original_tool is None:
            TOOLS.pop("test_tool", None)
        else:
            TOOLS["test_tool"] = original_tool

    assert result.success is False
    assert result.error == "Simulated failure"


def test_agent_builds_calculator_final_response():
    response = Agent._build_final_response(
        "calculator",
        ToolResult(
            success=True,
            tool="calculator",
            result=100,
        ),
    )

    assert isinstance(response, FinalResponse)
    assert response.answer == "100"
    assert response.tool == "calculator"
    assert response.tool_result == 100


def test_agent_formats_pdf_page_count():
    response = Agent._build_final_response(
        "pdf",
        ToolResult(
            success=True,
            tool="pdf",
            result={"pages": 2},
        ),
    )

    assert response.answer == "The PDF contains 2 pages."
    assert response.tool == "pdf"


def test_agent_formats_tool_failure():
    response = Agent._build_final_response(
        "calculator",
        ToolResult(
            success=False,
            tool="calculator",
            error="division by zero",
        ),
    )

    assert "division by zero" in response.answer
    assert response.tool == "calculator"
    assert response.tool_result == "division by zero"
