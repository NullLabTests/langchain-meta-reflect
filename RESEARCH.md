# Research: Meta-Reflective Agent with Procedural Memory

## Summary

This feature introduces a **self-improving agent** that learns across task executions
by maintaining a **procedural memory** of reusable heuristics distilled from its
own experience. It is implemented as a natural extension of LangChain's agent
and memory abstractions.

---

## Papers Surveyed

| # | Paper | arXiv | Year | Core Idea | Score (1-5) |
|---|-------|-------|------|-----------|-------------|
| 1 | **HyperAgents** | [2603.19461](https://arxiv.org/abs/2603.19461) | 2026 | Self-referential agent that modifies both its task policy *and* its self-modification logic | 5 |
| 2 | **Experiential Reflective Learning (ERL)** | [2603.24639](https://arxiv.org/abs/2603.24639) | 2026 | Reusable heuristics extracted from trajectories via LLM reflection and retrieved at test time | 5 |
| 3 | **Darwin Gödel Machine** | [2505.22954](https://arxiv.org/abs/2505.22954) | 2025 | Open-ended evolution of self-improving coding agents with an archive + mutation loop | 4 |
| 4 | **MARS (Metacognitive Agent Reflective Self-improvement)** | [2601.11974](https://arxiv.org/abs/2601.11974) | 2026 | Single-cycle self-evolution using principle-based + procedural reflection | 4 |
| 5 | **Mem^p: Exploring Agent Procedural Memory** | [2508.06433](https://arxiv.org/abs/2508.06433) | 2025 | Learnable, updatable procedural memory with build/retrieval/update strategies | 4 |
| 6 | **Recursive Agent Optimization (RAO)** | [2605.06639](https://arxiv.org/abs/2605.06639) | 2026 | Training a single LLM policy across recursively-generated execution trees | 4 |
| 7 | **Polaris** | [2603.23129](https://arxiv.org/abs/2603.23129) | 2026 | Recursive self-improvement for small models via experience abstraction + code patching | 3 |
| 8 | **AdMem** | [2606.06787](https://arxiv.org/abs/2606.06787) | 2026 | Unified semantic+episodic+procedural memory with multi-agent architecture | 3 |

## Scoring Methodology

Each paper was evaluated on five axes:

- **Novelty**: Is the concept genuinely new or a reapplication of existing ideas?
- **Feasibility**: Can it be implemented cleanly within LangChain's existing abstractions without deep surgery?
- **Composability**: Does the idea decompose into modular, reusable components that fit LangChain's philosophy?
- **Alignment**: Does it follow LangChain's design principles (LCEL, runnables, chain composition)?
- **Impact**: Would the feature meaningfully improve real-world agent systems?

## Feature Selection: Why Procedural Memory + Self-Improvement?

After deep analysis, I chose to implement a **Meta-Reflective Agent with
Procedural Memory** — a fusion of insights from ERL (reflective heuristic
extraction), HyperAgents (self-referential improvement), MARS (efficient
self-evolution), and Mem^p (procedural memory as a first-class object).

### The Core Insight

Current LangChain agents have **no mechanism to learn from experience across
task executions**. Each invocation starts from scratch. This means:

1. The same mistakes are repeated indefinitely.
2. Successful strategies are never captured or reused.
3. There is no "improvement over time" without manual prompt engineering.
4. Long-running agent systems remain brittle.

The research-backed solution is to add a **procedural memory** layer that:

1. **Captures** heuristics from execution trajectories via LLM-based reflection.
2. **Organizes** heuristics by type (success patterns, failure patterns, general
   guidelines) with rich metadata.
3. **Retrieves** relevant heuristics at inference time to guide the agent.
4. **Accumulates** knowledge across task executions, enabling genuine
   self-improvement.

### Why This Is Perfect for LangChain

- **Composable**: ProceduralMemory is a standalone abstraction that can be
  plugged into any agent. The SelfImprovingAgentExecutor wraps existing
  agents.
- **Follows existing patterns**: Mirrors BaseStore's key-value interface,
  uses ChatPromptTemplate, Runnable, and AgentExecutor conventions.
- **Zero breaking changes**: All existing agents continue to work unchanged.
- **Production value**: Self-improving agents reduce the need for manual
  prompt tuning, lower operational costs, and improve reliability.

---

## How the Feature Works

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              SelfImprovingAgentExecutor              │
│                                                      │
│  ┌──────────┐    ┌──────────────┐   ┌────────────┐  │
│  │  Input    │───▶│  Heuristic   │──▶│    Base    │  │
│  │  Task     │    │  Retrieval   │   │  LLM Agent │  │
│  └──────────┘    └──────────────┘   └──────┬─────┘  │
│                                            │         │
│  ┌──────────────────────────────┐          │         │
│  │   Execution Trajectory       │◀─────────┘         │
│  └──────────────┬───────────────┘                    │
│                 │                                    │
│  ┌──────────────▼───────────────┐                    │
│  │   Reflection & Heuristic     │                    │
│  │   Extraction (post-task)     │                    │
│  └──────────────┬───────────────┘                    │
│                 │                                    │
│  ┌──────────────▼───────────────┐                    │
│  │     Procedural Memory        │                    │
│  │  (Heuristic Store)           │                    │
│  └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

### Key Components

#### 1. `BaseProceduralMemory` (langchain_core.memory.procedural)
Abstract interface defining add/retrieve/get_all/clear operations over
`Heuristic` objects. Follows the same contract pattern as `BaseStore`.

#### 2. `InMemoryProceduralMemory` (langchain_classic.memory.procedural_memory)
Concrete implementation that uses an LLM to:
- **Reflect**: Given a task description + execution trace, extracts
  structured heuristics (SUCCESS_PATTERN, FAILURE_PATTERN, GENERAL_GUIDELINE).
- **Retrieve**: Given a new task description, scores candidate heuristics
  and returns the most relevant ones.

#### 3. `SelfImprovingAgentExecutor` (langchain_classic.agents.meta_reflect)
Wraps a tool-calling LLM agent with:
- **Pre-task injection**: Relevant heuristics are retrieved and injected into
  the agent's system prompt.
- **Post-task reflection**: After execution, the trajectory is reflected upon
  and new heuristics are stored.
- **Cross-task accumulation**: Heuristics persist across invocations.

---

## Comparison with Standard LangChain

| Aspect | Standard AgentExecutor | SelfImprovingAgentExecutor |
|--------|----------------------|---------------------------|
| Memory across runs | None (each run is independent) | Procedural memory accumulates experience |
| Mistake avoidance | Cannot learn from past mistakes | Failure patterns prevent recurrence |
| Strategy reuse | Manual prompt engineering | Automatic heuristic extraction |
| Improvement over time | Static | Self-improving with use |
| Complexity | Simple | Moderate (one extra component) |

---

## Usage Examples

```python
from langchain_openai import ChatOpenAI
from langchain_classic.agents.meta_reflect import create_meta_reflect_agent
from langchain_classic.memory.procedural_memory import (
    InMemoryProceduralMemory,
)

# Initialize the LLM and tools
llm = ChatOpenAI(model="gpt-4o")
tools = [ ... ]

# Create procedural memory (same LLM used for reflection)
memory = InMemoryProceduralMemory(llm=llm)

# Create self-improving agent
agent = create_meta_reflect_agent(
    llm=llm,
    tools=tools,
    procedural_memory=memory,
    verbose=True,
)

# First call — no prior experience
result1 = agent.invoke({"input": "Calculate 2^10"})
# Memory now contains heuristics about using calculator tools

# Second call — benefits from experience
result2 = agent.invoke({"input": "Calculate 3^5"})
# Agent will have relevant heuristics injected into its context

# Heuristics accumulate over time
print(f"Knowledge base: {len(agent.procedural_memory)} heuristics")
```

---

## Files Modified/Added

| File | Status | Purpose |
|------|--------|---------|
| `libs/core/langchain_core/memory/__init__.py` | **NEW** | Core memory package |
| `libs/core/langchain_core/memory/procedural.py` | **NEW** | Base abstractions (Heuristic, HeuristicType, BaseProceduralMemory) |
| `libs/langchain/langchain_classic/memory/procedural_memory.py` | **NEW** | InMemoryProceduralMemory (LLM-based reflection & retrieval) |
| `libs/langchain/langchain_classic/agents/meta_reflect/__init__.py` | **NEW** | Meta-reflect agent package |
| `libs/langchain/langchain_classic/agents/meta_reflect/base.py` | **NEW** | SelfImprovingAgentExecutor + create_meta_reflect_agent |
| `libs/langchain/langchain_classic/agents/meta_reflect/prompts.py` | **NEW** | Prompt templates |
| `libs/langchain/langchain_classic/agents/__init__.py` | **MODIFIED** | Exports for new agent |
| `libs/langchain/langchain_classic/memory/__init__.py` | **MODIFIED** | Exports for new memory |
| `libs/langchain/tests/unit_tests/agents/test_meta_reflect_core.py` | **NEW** | Core tests (10 tests) |

---

## References

1. Zhang et al. "HyperAgents: Self-Referential Agents that Modify Themselves."
   arXiv:2603.19461, 2026.
2. Experiential Reflective Learning. arXiv:2603.24639, 2026.
3. Zhang et al. "Darwin Gödel Machine: Open-Ended Evolution of
   Self-Improving Agents." arXiv:2505.22954, 2025.
4. MARS: Metacognitive Agent Reflective Self-improvement. arXiv:2601.11974,
   2026.
5. Mem^p: Exploring Agent Procedural Memory. arXiv:2508.06433, 2025.
6. Recursive Agent Optimization. arXiv:2605.06639, 2026.
7. Polaris: A Gödel Agent Framework for Small Language Models.
   arXiv:2603.23129, 2026.
8. AdMem: Advanced Memory for Task-solving Agents. arXiv:2606.06787, 2026.
