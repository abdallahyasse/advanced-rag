from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for all tools.

    Every tool must implement:
    - name
    - description
    - parameters
    - run()
    """

    name: str
    description: str
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the tool.
        """
        pass

    def schema(self) -> dict[str, Any]:
        """
        Return tool metadata used by the planner.
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }