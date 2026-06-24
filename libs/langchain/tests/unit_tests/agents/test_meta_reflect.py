"""Tests for the Meta-Reflective Agent with Procedural Memory."""

from langchain_core.memory.procedural import (
    BaseProceduralMemory,
    Heuristic,
    HeuristicType,
)
from langchain_classic.agents.meta_reflect.base import (
    SelfImprovingAgentExecutor,
    _build_heuristic_context,
    _format_trajectory,
)
from langchain_classic.memory.procedural_memory import (
    InMemoryProceduralMemory,
)


class FakeLLM:
    """A fake LLM that returns predetermined responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0

    def invoke(self, messages, config=None, **kwargs):
        self.call_count += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = "GENERAL_GUIDELINE: Think step by step."
        from langchain_core.messages import AIMessage
        return AIMessage(content=response)

    def bind_tools(self, tools):
        return self


class FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def invoke(self, input, config=None):
        from langchain_core.tools import Tool
        return Tool(name=self.name, description=self.description, func=lambda x: "42")


def test_heuristic_model() -> None:
    h = Heuristic(
        content="Always verify calculations twice.",
        type=HeuristicType.SUCCESS_PATTERN,
    )
    assert h.type == HeuristicType.SUCCESS_PATTERN
    assert "verify" in h.content
    assert h.id != ""


def test_procedural_memory_add_and_retrieve() -> None:
    llm = FakeLLM(responses=[
        "SUCCESS_PATTERN: Use a systematic approach.\n"
        "FAILURE_PATTERN: Avoid premature optimization.\n"
        "GENERAL_GUIDELINE: Verify intermediate results."
    ])
    memory = InMemoryProceduralMemory(llm=llm)

    # Test reflection and storage
    heuristics = memory.reflect_and_store(
        task_description="Calculate fibonacci numbers",
        trajectory="Thought: I should use recursion...\nObservation: Too slow",
    )
    assert len(heuristics) == 3
    assert len(memory) == 3

    # Test retrieval
    retrieved = memory.retrieve_heuristics(
        "Calculate factorial numbers",
        top_k=2,
    )
    assert len(retrieved) <= 2


def test_procedural_memory_empty_retrieval() -> None:
    llm = FakeLLM()
    memory = InMemoryProceduralMemory(llm=llm)
    assert memory.retrieve_heuristics("any task") == []


def test_procedural_memory_clear() -> None:
    llm = FakeLLM(responses=["GENERAL_GUIDELINE: Be careful."])
    memory = InMemoryProceduralMemory(llm=llm)
    memory.reflect_and_store("task", "trace")
    assert len(memory) == 1
    memory.clear()
    assert len(memory) == 0


def test_format_trajectory() -> None:
    from langchain_core.agents import AgentAction

    steps = [
        (AgentAction(tool="calculator", tool_input="2+2", log=""), "4"),
    ]
    result = _format_trajectory(steps)
    assert "Action: calculator" in result
    assert "Observation: 4" in result


def test_build_heuristic_context() -> None:
    heuristics = [
        Heuristic(content="Check your math.", type=HeuristicType.SUCCESS_PATTERN),
        Heuristic(content="Avoid infinite loops.", type=HeuristicType.FAILURE_PATTERN),
    ]
    context = _build_heuristic_context(heuristics, max_heuristics=5)
    assert "Check your math" in context
    assert "Avoid infinite loops" in context
    assert "Success Pattern" in context or "Success_Pattern" in context


def test_create_meta_reflect_agent() -> None:
    llm = FakeLLM(responses=[
        "GENERAL_GUIDELINE: Use tools wisely.",
        # Agent responses (for the executor loop)
        '{"output": "Final answer is 42."}',  # Will trigger AgentFinish
    ])
    memory = InMemoryProceduralMemory(llm=llm)
    tools = [FakeTool(name="calculator", description="Performs calculations")]

    executor = SelfImprovingAgentExecutor.from_llm_and_tools(
        llm=llm,
        tools=tools,
        procedural_memory=memory,
        reflect_after_each_run=False,
        verbose=False,
    )

    assert executor.procedural_memory is memory
    assert len(executor.procedural_memory) == 0  # No reflection happened yet


def test_heuristic_type_enum() -> None:
    assert HeuristicType.SUCCESS_PATTERN.value == "success_pattern"
    assert HeuristicType.FAILURE_PATTERN.value == "failure_pattern"
    assert HeuristicType.GENERAL_GUIDELINE.value == "general_guideline"
