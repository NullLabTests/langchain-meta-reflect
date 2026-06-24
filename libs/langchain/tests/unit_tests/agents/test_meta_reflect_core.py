"""Core-level tests for Meta-Reflective Agent -- no full langchain_classic deps."""

import sys
sys.path.insert(0, 'libs/core')
sys.path.insert(0, 'libs/core/langchain_core')

from langchain_core.memory.procedural import (
    BaseProceduralMemory,
    Heuristic,
    HeuristicType,
)

from collections.abc import Sequence


class SimpleProceduralMemory(BaseProceduralMemory):
    """Minimal in-memory implementation for testing."""

    def __init__(self):
        self._heuristics: list[Heuristic] = []

    def add_heuristics(self, heuristics: Sequence[Heuristic]) -> None:
        self._heuristics.extend(heuristics)

    def retrieve_heuristics(
        self, task_description: str, *, top_k: int = 5,
        types: Sequence[HeuristicType] | None = None,
    ) -> list[Heuristic]:
        candidates = self._heuristics
        if types:
            type_set = set(types)
            candidates = [h for h in candidates if h.type in type_set]
        return candidates[:top_k]

    def get_all_heuristics(self) -> list[Heuristic]:
        return list(self._heuristics)

    def clear(self) -> None:
        self._heuristics.clear()

    def __len__(self) -> int:
        return len(self._heuristics)


def test_heuristic_creation():
    h = Heuristic(content="Test heuristic", type=HeuristicType.SUCCESS_PATTERN)
    assert h.content == "Test heuristic"
    assert h.type == HeuristicType.SUCCESS_PATTERN
    assert isinstance(h.id, str)
    assert h.task_description == ""
    assert h.metadata == {}


def test_heuristic_defaults():
    h = Heuristic(content="Just a guideline")
    assert h.type == HeuristicType.GENERAL_GUIDELINE
    assert h.id == ""


def test_procedural_memory_add():
    mem = SimpleProceduralMemory()
    mem.add_heuristics([
        Heuristic(content="A", type=HeuristicType.SUCCESS_PATTERN),
        Heuristic(content="B", type=HeuristicType.FAILURE_PATTERN),
    ])
    assert len(mem) == 2


def test_procedural_memory_retrieve_all():
    mem = SimpleProceduralMemory()
    mem.add_heuristics([
        Heuristic(content="A", type=HeuristicType.SUCCESS_PATTERN),
        Heuristic(content="B", type=HeuristicType.FAILURE_PATTERN),
    ])
    result = mem.retrieve_heuristics("any", top_k=10)
    assert len(result) == 2


def test_procedural_memory_retrieve_filtered():
    mem = SimpleProceduralMemory()
    mem.add_heuristics([
        Heuristic(content="A", type=HeuristicType.SUCCESS_PATTERN),
        Heuristic(content="B", type=HeuristicType.FAILURE_PATTERN),
        Heuristic(content="C", type=HeuristicType.GENERAL_GUIDELINE),
    ])
    result = mem.retrieve_heuristics(
        "any", types=[HeuristicType.FAILURE_PATTERN]
    )
    assert len(result) == 1
    assert result[0].content == "B"


def test_procedural_memory_retrieve_top_k():
    mem = SimpleProceduralMemory()
    heuristics = [
        Heuristic(content=str(i)) for i in range(10)
    ]
    mem.add_heuristics(heuristics)
    result = mem.retrieve_heuristics("any", top_k=3)
    assert len(result) == 3


def test_procedural_memory_get_all():
    mem = SimpleProceduralMemory()
    assert mem.get_all_heuristics() == []
    mem.add_heuristics([Heuristic(content="X")])
    all_h = mem.get_all_heuristics()
    assert len(all_h) == 1
    assert all_h[0].content == "X"


def test_procedural_memory_clear():
    mem = SimpleProceduralMemory()
    mem.add_heuristics([Heuristic(content="X")])
    assert len(mem) == 1
    mem.clear()
    assert len(mem) == 0


def test_heuristic_type_values():
    assert HeuristicType.SUCCESS_PATTERN.value == "success_pattern"
    assert HeuristicType.FAILURE_PATTERN.value == "failure_pattern"
    assert HeuristicType.GENERAL_GUIDELINE.value == "general_guideline"


def test_heuristic_metadata():
    h = Heuristic(
        content="Test",
        metadata={"confidence": 0.9, "source": "reflection"},
    )
    assert h.metadata["confidence"] == 0.9
    assert h.metadata["source"] == "reflection"
