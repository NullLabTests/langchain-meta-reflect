"""Meta-Reflective Agent with Procedural Memory.

A self-improving agent that learns from past experiences by extracting,
storing, and retrieving reusable heuristics from execution trajectories.
"""

from langchain_classic.agents.meta_reflect.base import (
    SelfImprovingAgentExecutor,
    create_meta_reflect_agent,
)

__all__ = [
    "SelfImprovingAgentExecutor",
    "create_meta_reflect_agent",
]
