"""
Harvester v2 MCP Tool Servers

Model Context Protocol (MCP) servers providing tool access to sub-agents:

Servers:
    - scoring_mcp: Fraud scoring, churn prediction, propensity modeling
    - causality_mcp: Causal DAG, root cause analysis, what-if simulation
    - nba_mcp: Action scoring, script generation, campaign creation

Each MCP server exposes domain-specific tools that sub-agents can call
during their execution phase.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # MCP servers are standalone processes
