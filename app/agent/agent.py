from app.agent.executor import ToolExecutor
from app.agent.planner import Planner


class Agent:
    """
    Production Tool Calling Agent.
    """

    def __init__(self):

        self.planner = Planner()

        self.executor = ToolExecutor()

    def run(
        self,
        user_input: str,
    ):

        tool_call = self.planner.plan(user_input)

        print(f"\nThought : {tool_call.thought}")
        print(f"Tool     : {tool_call.tool}")
        print(f"Arguments: {tool_call.arguments}\n")

        result = self.executor.execute(
            tool_name=tool_call.tool,
            arguments=tool_call.arguments,
        )

        return result