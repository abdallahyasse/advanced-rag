from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for all tools.

    Every tool must implement:
    - name
    - description
    - run()
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the tool.
        """
        pass

    def schema(self) -> dict:
        """
        Return tool metadata used by the planner.
        """

        return {
            "name": self.name,
            "description": self.description,
        }