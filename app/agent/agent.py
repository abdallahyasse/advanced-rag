from app.agent.executor import ToolExecutor
from app.agent.models import FinalResponse, ToolResult
from app.agent.planner import Planner


class Agent:
    """
    Full LLM Tool Calling Agent.
    """

    def __init__(self):
        self.planner = Planner()
        self.executor = ToolExecutor()

    def run(self, user_input: str) -> FinalResponse:
        # Planning
        tool_call = self.planner.plan(user_input)

        print("\n========== Planning ==========")
        print(f"Thought   : {tool_call.thought}")
        print(f"Tool      : {tool_call.tool}")
        print(f"Arguments : {tool_call.arguments}")

        # Execute Tool
        result = self.executor.execute(
            tool_name=tool_call.tool,
            arguments=tool_call.arguments,
        )

        # Tool Result
        print("\n========== Tool Result ==========")

        if result.success:
            print(result.result)
        else:
            print(result.error)

        print("===============================\n")

        return self._build_final_response(
            tool_call.tool,
            result,
        )

    @staticmethod
    def _build_final_response(
        tool_name: str,
        result: ToolResult,
    ) -> FinalResponse:
        # Handle tool failure
        if not result.success:
            return FinalResponse(
                answer=(
                    f"The {tool_name} tool failed: "
                    f"{result.error}"
                ),
                tool=tool_name,
                tool_result=result.error,
            )

        tool_result = result.result

        # RAG Tool
        if tool_name == "rag":
            if isinstance(tool_result, dict):
                answer = tool_result.get(
                    "answer",
                    "I don't have enough information.",
                )
                return FinalResponse(
                    answer=str(answer),
                    tool=tool_name,
                    tool_result=tool_result,
                )

            return FinalResponse(
                answer=str(tool_result),
                tool=tool_name,
                tool_result=tool_result,
            )

        # Calculator Tool
        if tool_name == "calculator":
            return FinalResponse(
                answer=str(tool_result),
                tool=tool_name,
                tool_result=tool_result,
            )

        # PDF Tool
        if tool_name == "pdf":
            if isinstance(tool_result, dict):
                pages = tool_result.get("pages")
                page = tool_result.get("page")
                content = tool_result.get("content")

                if page is not None and content is not None:
                    answer = (
                        f"Content extracted from page {page}:\n"
                        f"{content}"
                    )
                elif pages is not None:
                    answer = f"The PDF contains {pages} pages."
                else:
                    answer = "PDF information retrieved successfully."

                return FinalResponse(
                    answer=answer,
                    tool=tool_name,
                    tool_result=tool_result,
                )

            return FinalResponse(
                answer=str(tool_result),
                tool=tool_name,
                tool_result=tool_result,
            )

        # Generic Tool
        return FinalResponse(
            answer=str(tool_result),
            tool=tool_name,
            tool_result=tool_result,
        )