from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory.procedural import (
    BaseProceduralMemory,
    Heuristic,
    HeuristicType,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import Field
from typing_extensions import override


_REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a meta-cognitive analyst reviewing an agent's execution trace. "
        "Extract concise, reusable heuristics (strategies, patterns, or rules) "
        "that would help the agent perform better on similar tasks in the future.\n\n"
        "Output exactly one heuristic per line. Each line must follow the format:\n"
        "TYPE: content\n\n"
        "Where TYPE is one of:\n"
        "- SUCCESS_PATTERN: a strategy that led to a good outcome\n"
        "- FAILURE_PATTERN: a mistake or pitfall to avoid\n"
        "- GENERAL_GUIDELINE: any other reusable rule of thumb\n\n"
        "If no useful heuristic can be extracted, output: NONE",
    ),
    ("human", "Task: {task_description}\n\nExecution Trace:\n{trace}"),
])


_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a heuristic retrieval system. Given a task description and "
        "a pool of candidate heuristics, select the ones most relevant to "
        "the current task. Return up to {top_k} heuristics, each on a "
        "separate line prefixed by its index. If none are relevant, "
        "output: NONE",
    ),
    ("human", "Task: {task_description}\n\nCandidate Heuristics:\n{candidates}"),
])


class InMemoryProceduralMemory(BaseProceduralMemory):
    """An in-memory procedural memory backed by an LLM for heuristic
    extraction and relevance scoring.

    This implementation:
    1. Uses an LLM to reflect on agent trajectories and extract heuristics.
    2. Stores heuristics in a list with optional embedding-based retrieval
       (via LLM relevance scoring if no embeddings are configured).
    3. Retrieves relevant heuristics at inference time.

    Examples:
        .. code-block:: python

            from langchain_classic.memory.procedural_memory import (
                InMemoryProceduralMemory,
            )
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o")
            memory = InMemoryProceduralMemory(llm=llm)

            # After a task execution, reflect and store heuristics
            memory.reflect_and_store(
                task_description="Calculate the sum of prime numbers up to 100",
                trajectory="Thought: I need to find primes...\\n"
                           "Action: calculator\\n...",
            )

            # Before a similar task, retrieve relevant heuristics
            heuristics = memory.retrieve_heuristics(
                task_description="Calculate the sum of even numbers up to 50",
                top_k=3,
            )
            for h in heuristics:
                print(f"  [{h.type.value}] {h.content}")
    """

    llm: BaseLanguageModel
    """The LLM used for reflection and retrieval."""

    heuristics: list[Heuristic] = Field(default_factory=list)
    """Internal storage for extracted heuristics."""

    reflection_prompt: ChatPromptTemplate = Field(default_factory=lambda: _REFLECTION_PROMPT)
    """Prompt template for extracting heuristics from execution traces."""

    retrieval_prompt: ChatPromptTemplate = Field(default_factory=lambda: _RETRIEVAL_PROMPT)
    """Prompt template for scoring and selecting relevant heuristics."""

    def _parse_heuristics(self, text: str, task_description: str) -> list[Heuristic]:
        lines = text.strip().splitlines()
        results: list[Heuristic] = []
        for line in lines:
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            if line.startswith("SUCCESS_PATTERN:"):
                content = line[len("SUCCESS_PATTERN:"):].strip()
                htype = HeuristicType.SUCCESS_PATTERN
            elif line.startswith("FAILURE_PATTERN:"):
                content = line[len("FAILURE_PATTERN:"):].strip()
                htype = HeuristicType.FAILURE_PATTERN
            elif line.startswith("GENERAL_GUIDELINE:"):
                content = line[len("GENERAL_GUIDELINE:"):].strip()
                htype = HeuristicType.GENERAL_GUIDELINE
            else:
                content = line
                htype = HeuristicType.GENERAL_GUIDELINE
            if content:
                results.append(
                    Heuristic(
                        id=str(uuid.uuid4()),
                        content=content,
                        type=htype,
                        task_description=task_description,
                    )
                )
        return results

    def reflect_and_store(
        self,
        task_description: str,
        trajectory: str,
        config: RunnableConfig | None = None,
    ) -> list[Heuristic]:
        """Reflect on an execution trajectory and store extracted heuristics.

        Args:
            task_description: Description of the task that was executed.
            trajectory: The execution trace (intermediate steps + observations).
            config: Optional runnable config.

        Returns:
            The list of newly extracted and stored heuristics.
        """
        messages = self.reflection_prompt.format_messages(
            task_description=task_description,
            trace=trajectory,
        )
        response = self.llm.invoke(messages, config=config)
        content = response.content if hasattr(response, "content") else str(response)
        heuristics = self._parse_heuristics(content, task_description)
        self.add_heuristics(heuristics)
        return heuristics

    @override
    def add_heuristics(self, heuristics: Sequence[Heuristic]) -> None:
        for h in heuristics:
            if not h.id:
                h.id = str(uuid.uuid4())
        self.heuristics.extend(heuristics)

    @override
    def retrieve_heuristics(
        self,
        task_description: str,
        *,
        top_k: int = 5,
        types: Sequence[HeuristicType] | None = None,
    ) -> list[Heuristic]:
        if not self.heuristics:
            return []

        candidates = self.heuristics
        if types:
            type_set = set(types)
            candidates = [h for h in candidates if h.type in type_set]

        if len(candidates) <= top_k:
            return list(candidates)

        candidate_text = "\n\n".join(
            f"[{i}] [{h.type.value}] {h.content}\n    "
            f"(from task: {h.task_description[:80]})"
            for i, h in enumerate(candidates)
        )
        messages = self.retrieval_prompt.format_messages(
            task_description=task_description,
            candidates=candidate_text,
            top_k=str(top_k),
        )
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        selected_indices: list[int] = []
        for line in content.strip().splitlines():
            line = line.strip()
            if line.upper() == "NONE":
                continue
            try:
                idx = int(line.split(":")[0].strip())
                if 0 <= idx < len(candidates):
                    selected_indices.append(idx)
            except (ValueError, IndexError):
                pass

        if not selected_indices:
            return candidates[:top_k]

        return [candidates[i] for i in selected_indices[:top_k]]

    @override
    def get_all_heuristics(self) -> list[Heuristic]:
        return list(self.heuristics)

    @override
    def clear(self) -> None:
        self.heuristics.clear()

    @override
    def __len__(self) -> int:
        return len(self.heuristics)
