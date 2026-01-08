"""
Autonomous Agents Module

This module contains autonomous agent implementations with internal sub-graphs,
as an alternative to the orchestrated agent pattern in the parent directory.

Key differences from orchestrated pattern:
- Each agent has its own compiled StateGraph with internal routing
- Agents make decisions about tool usage and execution flow
- More flexible but higher latency and cost
- Useful for complex multi-step workflows

Agents:
- AutonomousSearchAgent: Web search with adaptive querying
- AutonomousMemoryAgent: Memory storage and recall with classification
- AutonomousGenieAgent: Database queries with validation and review
"""

from .base_agent import AutonomousAgent
from .search_agent import AutonomousSearchAgent
from .memory_agent import AutonomousMemoryAgent
from .genie_agent import AutonomousGenieAgent

__all__ = [
    "AutonomousAgent",
    "AutonomousSearchAgent",
    "AutonomousMemoryAgent",
    "AutonomousGenieAgent",
]
