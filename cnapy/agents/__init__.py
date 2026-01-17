"""CNApy Multi-Agent System

This module provides a specialized sub-agent based system for natural language
analysis of metabolic models. It supports Korean/English auto-detection and
provides functionality for FBA, gene knockout analysis, scenario management,
and LLM-based strain knowledge queries.

Architecture:
    AgentDialog (Chat UI)
           │
           ▼
    OrchestratorAgent (Router)
           │
    ┌──────┼──────┬──────────┬──────────┬──────────┐
    │      │      │          │          │          │
    ▼      ▼      ▼          ▼          ▼          ▼
 Flux   Gene  Scenario    Data     Strain
Analysis Analysis Manager  Query   Knowledge
 Agent  Agent   Agent     Agent     Agent
"""

from cnapy.agents.base_agent import (
    AgentContext,
    AgentResponse,
    BaseAgent,
    Skill,
    SkillResult,
    ToolDefinition,
    WorkflowStep,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentContext",
    "SkillResult",
    "Skill",
    "ToolDefinition",
    "AgentResponse",
    "WorkflowStep",
]
