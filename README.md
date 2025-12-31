# Harvester V2 - Claude-Native Intelligence Engine

Production-level implementation of the DeepAgents-inspired intelligence engine for the Enterprise Agentization Platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│          CLAUDE-NATIVE HARVESTER V2 (DeepAgents-Inspired)               │
│                                                                          │
│  ORCHESTRATOR (LangGraph)                                               │
│  ├── Plans intelligence tasks (todo-style)                              │
│  ├── Delegates to sub-agents via asyncio.gather()                       │
│  ├── Aggregates results with LLM reasoning                              │
│  └── Critiques output quality (threshold 0.85)                          │
│                                                                          │
│     ┌────────────────────┬────────────────────┬────────────────────┐    │
│     │                    │                    │                    │    │
│     ▼                    ▼                    ▼                    ▼    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│  │ SCORING      │  │ CAUSALITY    │  │ NBA          │  │ REASONING│   │
│  │ SUB-AGENT    │  │ SUB-AGENT    │  │ SUB-AGENT    │  │ SUB-AGENT│   │
│  │ (isolated)   │  │ (isolated)   │  │ (isolated)   │  │(isolated)│   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘   │
│                                                                          │
│  MEMORY SYSTEM                                                          │
│  ├── Episodic Memory (Weaviate + PostgreSQL)                           │
│  ├── Semantic Retriever (RAG for domain knowledge)                     │
│  └── Playbook Selector (procedural memory)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **DeepAgents Pattern**: Plan → Execute → Critique → Refine lifecycle
- **Isolated Contexts**: Each sub-agent gets a fresh, immutable context
- **Parallel Execution**: Sub-agents run concurrently via `asyncio.gather()`
- **Phased Dependencies**: Dependent sub-agents run in sequence phases
- **Quality Control**: 0.85 critique threshold with up to 3 iterations
- **Multi-Tenant**: Full tenant isolation in all operations
- **Observability**: Langfuse tracing integration

## Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python --version  # Must be 3.11 or higher
   ```

2. **Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Ensure Docker daemon is running

3. **Required Python Packages**
   ```bash
   pip install pytest pytest-asyncio structlog weaviate-client asyncpg redis
   ```

## Quick Start

### 1. Start Database Services

```bash
# From project root
docker-compose -f harvester_v2/docker-compose.test.yml up -d

# Verify services are running
docker ps
```

**Services Started:**
| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5433 | Durable storage, learning tables |
| Weaviate | 8081 | Vector search, memory |
| Redis | 6380 | Caching |
| Kafka | 9093 | Event streaming (optional) |

### 2. Run Tests

```bash
# Set Python path
export PYTHONPATH=.

# Run unit tests (with mocks, no databases required)
pytest harvester_v2/tests/ -v -m "not integration and not e2e"

# Run integration tests (requires databases)
pytest harvester_v2/tests/ -v -m integration

# Run E2E tests (full pipeline)
pytest harvester_v2/tests/ -v -m e2e

# Run all tests
pytest harvester_v2/tests/ -v
```

### 3. Use Setup Script

```bash
# Check environment
python harvester_v2/scripts/setup_and_test.py --check

# Setup databases
python harvester_v2/scripts/setup_and_test.py --setup

# Apply migrations
python harvester_v2/scripts/setup_and_test.py --migrate

# Run all tests
python harvester_v2/scripts/setup_and_test.py --test-unit --test-int --test-e2e

# Do everything
python harvester_v2/scripts/setup_and_test.py --all
```

## Directory Structure

```
harvester_v2/
├── agents/
│   ├── base_subagent.py          # Base class with Plan→Execute→Critique
│   ├── parallel_executor.py       # asyncio.gather() orchestration
│   └── core/
│       ├── scoring_subagent.py    # Fraud/churn/propensity/risk
│       ├── causality_subagent.py  # Causal DAG, root cause, what-if
│       ├── nba_subagent.py        # Next best action with scripts
│       ├── reasoning_subagent.py  # Deep strategic reasoning
│       └── retainer_strategy_subagent.py  # Retention negotiation
├── memory/
│   ├── episodic_memory.py         # Cross-session memories
│   └── semantic_retriever.py      # RAG for domain knowledge
├── storage/
│   ├── weaviate_schema.py         # 4 Weaviate collections
│   └── postgresql_schema.sql      # 7 PostgreSQL tables
├── orchestrator/
│   └── intelligence_graph.py      # LangGraph workflow (7 stages)
├── tests/
│   ├── conftest.py                # Fixtures
│   ├── test_subagents.py          # Sub-agent unit tests
│   ├── test_orchestrator.py       # Orchestrator tests
│   └── test_e2e.py                # E2E tests
├── scripts/
│   └── setup_and_test.py          # Setup and test runner
├── docker-compose.test.yml        # Test environment
└── README.md                      # This file
```

## Sub-Agent Framework

### Creating a New Sub-Agent

```python
from harvester_v2.agents.base_subagent import BaseSubAgent, SubAgentContext

class MySubAgent(BaseSubAgent):
    """Custom sub-agent following DeepAgents pattern."""

    def __init__(self, options=None):
        super().__init__(agent_name="my_subagent", options=options)

    async def _plan(self, context: SubAgentContext, task: str) -> dict:
        """Generate execution plan."""
        # Use LLM to plan
        prompt = f"Plan how to: {task}\nContext: {context.to_prompt_context()}"
        response = await self.model_router.complete(prompt=prompt, ...)
        return json.loads(response)

    async def _execute_plan(self, context: SubAgentContext, plan: dict) -> dict:
        """Execute the plan."""
        # Your business logic here
        return {"result": "..."}
```

### Running Sub-Agents

```python
from harvester_v2.agents.base_subagent import SubAgentContext
from harvester_v2.agents.core.scoring_subagent import ScoringSubAgent

# Create context
context = SubAgentContext(
    tenant_id="gigaclear",
    customer_id="CUST-123",
    input_data={"product": "broadband_500", "tenure_months": 24}
)

# Run sub-agent
agent = ScoringSubAgent()
result = await agent.run_isolated(context, "Calculate risk scores")

if result.success:
    print(f"Scores: {result.data}")
    print(f"Critique: {result.critique_score}")
```

### Parallel Execution

```python
from harvester_v2.agents.parallel_executor import ParallelSubAgentExecutor

# Create executor with multiple agents
executor = ParallelSubAgentExecutor([
    ScoringSubAgent(),
    CausalitySubAgent(),
    ReasoningSubAgent()
])

# Execute all in parallel
result = await executor.execute_parallel(
    base_context=context,
    subagent_names=["scoring_subagent", "causality_subagent"],
    task_description="Analyze customer"
)

# Or execute in phases (dependencies)
phases = [
    ["scoring_subagent"],           # Phase 1: Scoring first
    ["causality_subagent", "reasoning_subagent"],  # Phase 2: Analysis
    ["nba_subagent"]                # Phase 3: NBA uses scores
]
result = await executor.execute_phased(context, phases, "Full analysis")
```

## Database Schemas

### Weaviate Collections

| Collection | Purpose |
|------------|---------|
| `EpisodicMemory` | Cross-session memories for entities |
| `SemanticKnowledge` | Domain knowledge for RAG |
| `CustomerIntelligenceV2` | Unified customer intelligence |
| `NextBestActionV2` | Generated NBA with outcomes |

### PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `episodic_memories` | Backup from Weaviate |
| `playbooks` | Procedural memory |
| `action_outcomes` | Reinforcement learning |
| `training_data` | Fine-tuning pairs |
| `execution_traces` | Observability |
| `subagent_metrics` | Performance metrics |
| `customer_intelligence_history` | Historical snapshots |

## Configuration

### Environment Variables

```bash
# Database connections
WEAVIATE_URL=http://localhost:8081
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
REDIS_URL=redis://localhost:6380

# LLM (via model_router)
ANTHROPIC_API_KEY=...

# Langfuse (optional)
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
```

### Tenant Configuration

See `config/tenants/gigaclear/agents.yaml` for tenant-specific sub-agent configuration.

## Troubleshooting

### Docker Not Running

```bash
# Check Docker status
docker info

# If not running, start Docker Desktop
```

### Database Connection Failed

```bash
# Check if services are up
docker ps

# Check logs
docker logs harvester_v2_postgres
docker logs harvester_v2_weaviate
```

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=.

# Or on Windows
set PYTHONPATH=.
```

### Tests Failing

```bash
# Run with verbose output
pytest harvester_v2/tests/ -v -s

# Run specific test
pytest harvester_v2/tests/test_subagents.py::TestScoringSubAgent -v
```

## Performance Metrics

| Metric | Target |
|--------|--------|
| Sub-agent execution | < 5s |
| Parallel speedup | 2-3x vs sequential |
| Critique threshold | 0.85 |
| Max iterations | 3 |

## Contributing

1. Follow DeepAgents pattern for new sub-agents
2. Ensure all tests pass
3. Add tests for new functionality
4. Update this README if adding new features

## License

Proprietary - nexgAI
