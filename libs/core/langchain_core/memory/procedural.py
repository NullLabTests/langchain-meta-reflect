from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HeuristicType(str, Enum):
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    GENERAL_GUIDELINE = "general_guideline"


class Heuristic(BaseModel):
    id: str = Field(default="", description="Unique identifier for this heuristic.")
    content: str = Field(description="The heuristic text (instruction, strategy, or rule).")
    type: HeuristicType = Field(
        default=HeuristicType.GENERAL_GUIDELINE,
        description="Classification of this heuristic.",
    )
    task_description: str = Field(
        default="",
        description="The task description from which this heuristic was extracted.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., confidence, success_count, failure_count).",
    )


class BaseProceduralMemory(ABC):
    """Abstract interface for procedural memory that stores and retrieves
    reusable heuristics distilled from past agent trajectories.

    Procedural memory captures *how* to do things — strategies, patterns,
    and rules of thumb — rather than raw conversational history. This
    abstraction follows the neuro-symbolic memory paradigm described in
    Experiential Reflective Learning (ERL, arXiv:2603.24639) and
    HyperAgents (arXiv:2603.19461).
    """

    @abstractmethod
    def add_heuristics(self, heuristics: Sequence[Heuristic]) -> None:
        """Store one or more heuristics into procedural memory.

        Args:
            heuristics: The heuristics to persist.
        """
        ...

    @abstractmethod
    def retrieve_heuristics(
        self,
        task_description: str,
        *,
        top_k: int = 5,
        types: Sequence[HeuristicType] | None = None,
    ) -> list[Heuristic]:
        """Retrieve the most relevant heuristics for a given task.

        Args:
            task_description: The task to find relevant heuristics for.
            top_k: Maximum number of heuristics to return.
            types: Optional filter by heuristic type(s).

        Returns:
            A list of heuristics ordered by estimated relevance.
        """
        ...

    @abstractmethod
    def get_all_heuristics(self) -> list[Heuristic]:
        """Return every heuristic currently stored in memory.

        Returns:
            All heuristics in no particular order.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all heuristics from procedural memory."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored heuristics."""
        ...
