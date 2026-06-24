from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory.procedural import BaseProceduralMemory, Heuristic, HeuristicType
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.meta_reflect.prompts import (
    AGENT_SYSTEM_PROMPT,
    HEURISTIC_CONTEXT_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
)
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser

_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. Use them wisely.

{heuristic_context}
"""


class MetaReflectInput(BaseModel):
    input: str = Field(description="The user's input or task description.")
    chat_history: list[BaseMessage] = Field(
        default_factory=list,
        description="Conversation history.",
    )
    intermediate_steps: list[tuple[AgentAction, str]] = Field(
        default_factory=list,
        description="Steps taken so far by the agent executor.",
    )


def _format_tools(tools: Sequence[BaseTool]) -> str:
    lines: list[str] = []
    for t in tools:
        desc = t.description or "No description"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def _build_heuristic_context(
    heuristics: list[Heuristic],
    max_heuristics: int = 5,
) -> str:
    if not heuristics:
        return ""
    selected = heuristics[:max_heuristics]
    parts: list[str] = []
    for i, h in enumerate(selected, 1):
        tag = h.type.value.replace("_", " ").title()
        parts.append(f"{i}. [{tag}] {h.content}")
    return HEURISTIC_CONTEXT_PROMPT.format(
        heuristics_text="\n".join(parts),
    )


def _format_trajectory(intermediate_steps: list[tuple[AgentAction, str]]) -> str:
    lines: list[str] = []
    for action, observation in intermediate_steps:
        lines.append(f"Action: {action.tool}({action.tool_input})")
        lines.append(f"Observation: {str(observation)[:200]}")
    return "\n".join(lines) if lines else "(no actions taken)"


class SelfImprovingAgentExecutor:
    """An agent executor that augments a base agent with procedural memory
    for self-improvement across task executions.

    Before each invocation, relevant heuristics from past executions are
    retrieved from procedural memory and injected into the agent's context.
    After each invocation, the agent reflects on its execution trajectory
    and extracts new heuristics to store for future use.

    This implements a reflective self-improvement loop inspired by:
    - Experiential Reflective Learning (arXiv:2603.24639)
    - HyperAgents (arXiv:2603.19461)
    - MARS (arXiv:2601.11974)

    Examples:
        .. code-block:: python

            from langchain_classic.agents.meta_reflect.base import (
                SelfImprovingAgentExecutor,
            )
            from langchain_classic.memory.procedural_memory import (
                InMemoryProceduralMemory,
            )
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o")
            tools = [...]
            memory = InMemoryProceduralMemory(llm=llm)

            executor = SelfImprovingAgentExecutor.from_llm_and_tools(
                llm=llm,
                tools=tools,
                procedural_memory=memory,
            )

            # First invocation
            result = executor.invoke({"input": "Calculate 25 * 4 + 10"})

            # Second invocation — automatically benefits from past heuristics
            result = executor.invoke({"input": "Calculate (15 + 7) * 3"})
    """

    def __init__(
        self,
        agent: Runnable,
        tools: Sequence[BaseTool],
        procedural_memory: BaseProceduralMemory,
        *,
        reflect_after_each_run: bool = True,
        max_heuristics_per_run: int = 5,
        verbose: bool = False,
    ) -> None:
        self._agent = agent
        self._tools = list(tools)
        self._tools_map = {t.name: t for t in tools}
        self._memory = procedural_memory
        self._reflect = reflect_after_each_run
        self._max_heuristics = max_heuristics_per_run
        self._verbose = verbose
        self._trajectory_history: dict[str, list[tuple[AgentAction, str]]] = {}

    @property
    def procedural_memory(self) -> BaseProceduralMemory:
        return self._memory

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        procedural_memory: BaseProceduralMemory,
        *,
        system_prompt: str | None = None,
        reflect_after_each_run: bool = True,
        max_heuristics_per_run: int = 5,
        verbose: bool = False,
    ) -> SelfImprovingAgentExecutor:
        """Create a self-improving agent executor from an LLM and tools.

        Args:
            llm: The language model to use.
            tools: The tools available to the agent.
            procedural_memory: The procedural memory store.
            system_prompt: Optional custom system prompt. Must contain
                ``{heuristic_context}`` if provided.
            reflect_after_each_run: Whether to reflect after each invocation.
            max_heuristics_per_run: Maximum heuristics to inject per run.
            verbose: Whether to print detailed logs.

        Returns:
            A configured SelfImprovingAgentExecutor.
        """
        if not hasattr(llm, "bind_tools"):
            msg = "LLM must support bind_tools() for tool calling."
            raise TypeError(msg)

        prompt_text = system_prompt or _DEFAULT_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm_with_tools = llm.bind_tools(tools)

        agent: Runnable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "heuristic_context": lambda x: x.get("heuristic_context", ""),
                "agent_scratchpad": lambda x: format_to_tool_messages(
                    x.get("intermediate_steps", [])
                ),
            }
            | prompt
            | llm_with_tools
            | ToolsAgentOutputParser()
        )

        return cls(
            agent=agent,
            tools=tools,
            procedural_memory=procedural_memory,
            reflect_after_each_run=reflect_after_each_run,
            max_heuristics_per_run=max_heuristics_per_run,
            verbose=verbose,
        )

    def _retrieve_and_inject_heuristics(
        self,
        task_description: str,
    ) -> str:
        heuristics = self._memory.retrieve_heuristics(
            task_description,
            top_k=self._max_heuristics,
        )
        context = _build_heuristic_context(heuristics, self._max_heuristics)
        if self._verbose and context:
            print(f"[MetaReflect] Injected {len(heuristics)} heuristics")
        return context

    def _reflect_on_trajectory(
        self,
        task_description: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        config: RunnableConfig | None = None,
    ) -> list[Heuristic]:
        trajectory = _format_trajectory(intermediate_steps)
        if self._verbose:
            print(f"[MetaReflect] Reflecting on task: {task_description[:60]}...")
        heuristics = self._memory.reflect_and_store(
            task_description=task_description,
            trajectory=trajectory,
            config=config,
        )
        if self._verbose and heuristics:
            print(f"[MetaReflect] Stored {len(heuristics)} new heuristics")
        return heuristics

    def invoke(
        self,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        task_description = inputs.get("input", "")
        heuristic_context = self._retrieve_and_inject_heuristics(task_description)

        run_inputs = {
            "input": task_description,
            "heuristic_context": heuristic_context,
            "chat_history": inputs.get("chat_history", []),
            "intermediate_steps": [],
        }

        max_iterations = kwargs.get("max_iterations", 15)
        early_stopping_method = kwargs.get("early_stopping_method", "force")

        steps: list[tuple[AgentAction, str]] = []
        final_output: dict[str, Any] = {"output": ""}

        for iteration in range(max_iterations):
            run_inputs["intermediate_steps"] = steps
            try:
                output = self._agent.invoke(run_inputs, config=config)
            except Exception as e:
                if self._verbose:
                    print(f"[MetaReflect] Agent error at iteration {iteration}: {e}")
                break

            if isinstance(output, AgentFinish):
                final_output = {"output": output.return_values.get("output", "")}
                break
            elif isinstance(output, AgentAction):
                tool = self._tools_map.get(output.tool)
                if tool is None:
                    observation = f"Error: unknown tool '{output.tool}'"
                else:
                    try:
                        tool_result = tool.invoke(output.tool_input, config=config)
                        observation = (
                            str(tool_result.content)
                            if hasattr(tool_result, "content")
                            else str(tool_result)
                        )
                    except Exception as e:
                        observation = f"Error: {e!s}"
                steps.append((output, observation))
            else:
                final_output = {"output": str(output)}
                break
        else:
            final_output = {"output": final_output.get("output", "")}

        if self._reflect:
            self._reflect_on_trajectory(task_description, steps, config=config)

        return final_output

    async def ainvoke(
        self,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        task_description = inputs.get("input", "")
        heuristic_context = self._retrieve_and_inject_heuristics(task_description)

        run_inputs = {
            "input": task_description,
            "heuristic_context": heuristic_context,
            "chat_history": inputs.get("chat_history", []),
            "intermediate_steps": [],
        }

        max_iterations = kwargs.get("max_iterations", 15)
        steps: list[tuple[AgentAction, str]] = []
        final_output: dict[str, Any] = {"output": ""}

        for iteration in range(max_iterations):
            run_inputs["intermediate_steps"] = steps
            try:
                output = await self._agent.ainvoke(run_inputs, config=config)
            except Exception as e:
                if self._verbose:
                    print(f"[MetaReflect] Agent error at iteration {iteration}: {e}")
                break

            if isinstance(output, AgentFinish):
                final_output = {"output": output.return_values.get("output", "")}
                break
            elif isinstance(output, AgentAction):
                tool = self._tools_map.get(output.tool)
                if tool is None:
                    observation = f"Error: unknown tool '{output.tool}'"
                else:
                    try:
                        tool_result = await tool.ainvoke(output.tool_input, config=config)
                        observation = (
                            str(tool_result.content)
                            if hasattr(tool_result, "content")
                            else str(tool_result)
                        )
                    except Exception as e:
                        observation = f"Error: {e!s}"
                steps.append((output, observation))
            else:
                final_output = {"output": str(output)}
                break
        else:
            final_output = {"output": final_output.get("output", "")}

        if self._reflect:
            await self._memory.reflect_and_store(
                task_description=task_description,
                trajectory=_format_trajectory(steps),
                config=config,
            )

        return final_output


def create_meta_reflect_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    procedural_memory: BaseProceduralMemory,
    *,
    system_prompt: str | None = None,
    reflect_after_each_run: bool = True,
    max_heuristics_per_run: int = 5,
    verbose: bool = False,
) -> SelfImprovingAgentExecutor:
    """Create a self-improving agent that learns from experience across tasks.

    This agent combines:
    - A tool-calling LLM agent for task execution
    - Procedural memory for storing and retrieving reusable heuristics
    - A reflective loop that extracts lessons from each execution

    Args:
        llm: The language model.
        tools: The tools available.
        procedural_memory: Memory store for heuristics.
        system_prompt: Optional custom system prompt.
        reflect_after_each_run: Whether to reflect post-execution.
        max_heuristics_per_run: Max heuristics to inject before each run.
        verbose: Print detailed logs.

    Returns:
        A SelfImprovingAgentExecutor ready to invoke.
    """
    return SelfImprovingAgentExecutor.from_llm_and_tools(
        llm=llm,
        tools=tools,
        procedural_memory=procedural_memory,
        system_prompt=system_prompt,
        reflect_after_each_run=reflect_after_each_run,
        max_heuristics_per_run=max_heuristics_per_run,
        verbose=verbose,
    )
