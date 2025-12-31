-- ============================================================================
-- Harvester V2 PostgreSQL Schema
-- ============================================================================
-- Learning tables for the Claude-native intelligence engine
--
-- Tables:
-- 1. episodic_memories - Backup from Weaviate (for durability)
-- 2. playbooks - Procedural memory (success patterns)
-- 3. action_outcomes - Reinforcement learning data
-- 4. training_data - DPO pairs for fine-tuning
-- 5. execution_traces - Observability data
-- 6. subagent_metrics - Sub-agent performance metrics
--
-- Architecture:
-- - PostgreSQL is the durable store (Weaviate may lose data)
-- - All learning data flows through PostgreSQL first
-- - Weaviate is used for vector search only
-- ============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS harvester_v2;

-- ============================================================================
-- 1. Episodic Memories (backup from Weaviate)
-- ============================================================================
-- Stores cross-session memories for durability
-- Weaviate stores the same data with vectors for semantic search

CREATE TABLE IF NOT EXISTS harvester_v2.episodic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Entity identification
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,  -- customer, session, agent
    memory_type VARCHAR(50) NOT NULL,  -- interaction, preference, outcome, insight

    -- Memory content
    content TEXT NOT NULL,
    context_summary TEXT,
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),

    -- Provenance
    tenant_id VARCHAR(50) NOT NULL,
    session_id VARCHAR(255),
    agent_name VARCHAR(100),

    -- Weaviate sync
    weaviate_id VARCHAR(255),  -- UUID from Weaviate for sync
    synced_to_weaviate BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Indexes
    CONSTRAINT idx_episodic_entity_unique UNIQUE (entity_id, memory_type, tenant_id, created_at)
);

-- Indexes for episodic_memories
CREATE INDEX IF NOT EXISTS idx_episodic_entity ON harvester_v2.episodic_memories(entity_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_type ON harvester_v2.episodic_memories(memory_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_importance ON harvester_v2.episodic_memories(importance DESC, tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_created ON harvester_v2.episodic_memories(created_at DESC, tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_weaviate_sync ON harvester_v2.episodic_memories(synced_to_weaviate) WHERE synced_to_weaviate = FALSE;

-- ============================================================================
-- 2. Playbooks (procedural memory)
-- ============================================================================
-- Success playbooks learned from outcomes
-- Used by PlaybookSelector to match situations to proven patterns

CREATE TABLE IF NOT EXISTS harvester_v2.playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Playbook identification
    playbook_id VARCHAR(100) UNIQUE NOT NULL,
    playbook_name VARCHAR(255) NOT NULL,

    -- Trigger conditions (when to apply this playbook)
    trigger_conditions JSONB NOT NULL,
    -- Example: {"churn_risk_min": 0.6, "segment": "high_value", "tenure_months_min": 12}

    -- Steps to execute
    steps JSONB NOT NULL,
    -- Example: [{"step": 1, "action": "offer_discount", "params": {"percent": 20}}, ...]

    -- Success metrics
    success_rate FLOAT DEFAULT 0.0 CHECK (success_rate >= 0 AND success_rate <= 1),
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    avg_reward FLOAT DEFAULT 0.0,

    -- Scope
    tenant_id VARCHAR(50) NOT NULL,
    domain VARCHAR(50),  -- telecom, banking, insurance
    action_type VARCHAR(50),  -- retention, upsell, cross_sell

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    min_confidence FLOAT DEFAULT 0.7,  -- Min confidence to use this playbook

    -- Timestamps
    last_used TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for playbooks
CREATE INDEX IF NOT EXISTS idx_playbooks_tenant ON harvester_v2.playbooks(tenant_id, domain);
CREATE INDEX IF NOT EXISTS idx_playbooks_action_type ON harvester_v2.playbooks(action_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_success ON harvester_v2.playbooks(success_rate DESC, tenant_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_playbooks_active ON harvester_v2.playbooks(is_active, tenant_id);

-- ============================================================================
-- 3. Action Outcomes (reinforcement learning)
-- ============================================================================
-- Records outcomes of actions taken for learning
-- Used to train reward models and update playbook success rates

CREATE TABLE IF NOT EXISTS harvester_v2.action_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Action identification
    action_id VARCHAR(100) NOT NULL,
    nba_id VARCHAR(100),  -- Link to NextBestActionV2 in Weaviate
    playbook_id VARCHAR(100),  -- Which playbook was used (if any)

    -- Context
    nba_type VARCHAR(50) NOT NULL,  -- retention, upsell, cross_sell, service
    customer_id VARCHAR(255) NOT NULL,

    -- Action details
    action_taken JSONB NOT NULL,
    -- Example: {"action": "offer_discount", "discount_percent": 20, "product": "premium_plan"}

    -- Features at time of action (for learning)
    features JSONB,
    -- Example: {"churn_risk": 0.7, "ltv": 1500, "tenure_months": 24, "segment": "high_value"}

    -- Predicted values
    predicted_confidence FLOAT,
    predicted_value FLOAT,

    -- Outcome
    outcome VARCHAR(50) NOT NULL,  -- accepted, rejected, ignored, partial, expired
    outcome_details JSONB,
    -- Example: {"accepted_on": "2025-01-15", "product_selected": "premium_plan", "contract_months": 12}

    -- Reward calculation
    reward FLOAT,  -- Computed reward signal
    actual_value FLOAT,  -- Actual monetary impact

    -- Reasoning
    model_used VARCHAR(50),
    reasoning_trace TEXT,

    -- Provenance
    tenant_id VARCHAR(50) NOT NULL,
    orchestrator_run_id VARCHAR(100),

    -- Timestamps
    action_created_at TIMESTAMPTZ,
    outcome_recorded_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for action_outcomes
CREATE INDEX IF NOT EXISTS idx_outcomes_customer ON harvester_v2.action_outcomes(customer_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_nba_type ON harvester_v2.action_outcomes(nba_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON harvester_v2.action_outcomes(outcome, tenant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_playbook ON harvester_v2.action_outcomes(playbook_id, tenant_id) WHERE playbook_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_outcomes_reward ON harvester_v2.action_outcomes(reward DESC, tenant_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_created ON harvester_v2.action_outcomes(created_at DESC, tenant_id);

-- ============================================================================
-- 4. Training Data (for fine-tuning)
-- ============================================================================
-- Stores DPO pairs and supervised fine-tuning data
-- Generated from action outcomes with positive/negative examples

CREATE TABLE IF NOT EXISTS harvester_v2.training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Training data type
    data_type VARCHAR(50) NOT NULL,  -- dpo, sft, preference, reward

    -- The prompt/context
    prompt TEXT NOT NULL,

    -- Responses (for DPO: chosen vs rejected)
    chosen_response TEXT NOT NULL,
    rejected_response TEXT,  -- For DPO pairs

    -- Context and features
    context JSONB,
    -- Example: {"customer_segment": "high_value", "churn_risk": 0.7, "products": ["broadband"]}

    -- Reward signals
    reward_signal FLOAT,
    chosen_reward FLOAT,
    rejected_reward FLOAT,

    -- Source tracking
    source_action_id VARCHAR(100),  -- Link to action_outcomes
    source_outcome_id UUID,  -- Link to action_outcomes.id

    -- Training status
    used_for_training BOOLEAN DEFAULT FALSE,
    training_batch_id VARCHAR(100),
    training_timestamp TIMESTAMPTZ,

    -- Quality
    quality_score FLOAT,  -- Human or automated quality rating
    validated_by VARCHAR(100),  -- Who validated this data

    -- Scope
    tenant_id VARCHAR(50) NOT NULL,
    domain VARCHAR(50),  -- telecom, banking, insurance
    task_type VARCHAR(50),  -- nba_generation, scoring, reasoning

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for training_data
CREATE INDEX IF NOT EXISTS idx_training_unused ON harvester_v2.training_data(used_for_training, tenant_id) WHERE used_for_training = FALSE;
CREATE INDEX IF NOT EXISTS idx_training_type ON harvester_v2.training_data(data_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_training_task ON harvester_v2.training_data(task_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_training_quality ON harvester_v2.training_data(quality_score DESC, tenant_id);
CREATE INDEX IF NOT EXISTS idx_training_created ON harvester_v2.training_data(created_at DESC, tenant_id);

-- ============================================================================
-- 5. Execution Traces (observability)
-- ============================================================================
-- Records execution traces for debugging and analysis
-- Complements Langfuse with PostgreSQL durability

CREATE TABLE IF NOT EXISTS harvester_v2.execution_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Trace identification
    trace_id VARCHAR(100) NOT NULL,
    span_id VARCHAR(100),
    parent_span_id VARCHAR(100),

    -- Run identification
    orchestrator_run_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(255),

    -- Sub-agent identification
    subagent_name VARCHAR(100) NOT NULL,
    stage VARCHAR(50),  -- plan, execute, critique, refine

    -- Input/Output
    input_state JSONB,
    output_state JSONB,

    -- Performance
    duration_ms INT,
    model_used VARCHAR(50),
    token_count INT,
    prompt_tokens INT,
    completion_tokens INT,

    -- Quality
    critique_score FLOAT,
    iteration_count INT DEFAULT 1,

    -- Error tracking
    error_message TEXT,
    error_type VARCHAR(100),
    stack_trace TEXT,

    -- Langfuse sync
    langfuse_trace_id VARCHAR(100),
    langfuse_span_id VARCHAR(100),
    synced_to_langfuse BOOLEAN DEFAULT FALSE,

    -- Scope
    tenant_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(255),

    -- Timestamps
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for execution_traces
CREATE INDEX IF NOT EXISTS idx_traces_run ON harvester_v2.execution_traces(orchestrator_run_id);
CREATE INDEX IF NOT EXISTS idx_traces_subagent ON harvester_v2.execution_traces(subagent_name, tenant_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON harvester_v2.execution_traces(session_id, tenant_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_traces_error ON harvester_v2.execution_traces(error_type, tenant_id) WHERE error_message IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_traces_langfuse_sync ON harvester_v2.execution_traces(synced_to_langfuse) WHERE synced_to_langfuse = FALSE;
CREATE INDEX IF NOT EXISTS idx_traces_created ON harvester_v2.execution_traces(created_at DESC, tenant_id);

-- ============================================================================
-- 6. Sub-Agent Metrics (performance tracking)
-- ============================================================================
-- Aggregated metrics for sub-agent performance
-- Used for monitoring and optimization

CREATE TABLE IF NOT EXISTS harvester_v2.subagent_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Sub-agent identification
    subagent_name VARCHAR(100) NOT NULL,

    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    period_type VARCHAR(20) NOT NULL,  -- hourly, daily, weekly

    -- Execution metrics
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    success_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN execution_count > 0 THEN success_count::FLOAT / execution_count ELSE 0 END
    ) STORED,

    -- Performance metrics
    avg_duration_ms FLOAT,
    p50_duration_ms FLOAT,
    p95_duration_ms FLOAT,
    p99_duration_ms FLOAT,

    -- Token usage
    total_tokens INT DEFAULT 0,
    avg_tokens_per_execution FLOAT,

    -- Quality metrics
    avg_critique_score FLOAT,
    avg_iterations FLOAT,
    refinement_rate FLOAT,  -- % of executions that needed refinement

    -- Cost (estimated)
    estimated_cost_usd FLOAT,

    -- Scope
    tenant_id VARCHAR(50) NOT NULL,
    model_used VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint
    CONSTRAINT idx_subagent_metrics_unique UNIQUE (subagent_name, period_start, period_type, tenant_id)
);

-- Indexes for subagent_metrics
CREATE INDEX IF NOT EXISTS idx_metrics_subagent ON harvester_v2.subagent_metrics(subagent_name, tenant_id);
CREATE INDEX IF NOT EXISTS idx_metrics_period ON harvester_v2.subagent_metrics(period_start DESC, period_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_metrics_success ON harvester_v2.subagent_metrics(success_rate DESC, tenant_id);

-- ============================================================================
-- 7. Customer Intelligence Snapshots (versioned history)
-- ============================================================================
-- Historical snapshots of customer intelligence for analysis
-- Enables "what changed" and "when did we know" queries

CREATE TABLE IF NOT EXISTS harvester_v2.customer_intelligence_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Customer identification
    customer_id VARCHAR(255) NOT NULL,

    -- Scores at snapshot time
    fraud_score FLOAT,
    churn_risk FLOAT,
    propensity_score FLOAT,
    risk_score FLOAT,
    engagement_score FLOAT,

    -- Classification
    segment VARCHAR(50),
    ltv FLOAT,

    -- Analysis results
    causal_factors JSONB,
    strategic_insights JSONB,
    reasoning_summary TEXT,

    -- Run that generated this
    orchestrator_run_id VARCHAR(100),
    model_version VARCHAR(50),

    -- Scope
    tenant_id VARCHAR(50) NOT NULL,

    -- Weaviate reference
    weaviate_id VARCHAR(255),

    -- Timestamps
    snapshot_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Indexes for customer_intelligence_history
CREATE INDEX IF NOT EXISTS idx_ci_history_customer ON harvester_v2.customer_intelligence_history(customer_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_ci_history_snapshot ON harvester_v2.customer_intelligence_history(snapshot_at DESC, tenant_id);
CREATE INDEX IF NOT EXISTS idx_ci_history_segment ON harvester_v2.customer_intelligence_history(segment, tenant_id);
CREATE INDEX IF NOT EXISTS idx_ci_history_churn ON harvester_v2.customer_intelligence_history(churn_risk DESC, tenant_id);

-- ============================================================================
-- Functions
-- ============================================================================

-- Function to update playbook success rates
CREATE OR REPLACE FUNCTION harvester_v2.update_playbook_success_rate(p_playbook_id VARCHAR)
RETURNS VOID AS $$
DECLARE
    v_total INT;
    v_success INT;
    v_avg_reward FLOAT;
BEGIN
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE outcome IN ('accepted', 'partial')),
        AVG(reward)
    INTO v_total, v_success, v_avg_reward
    FROM harvester_v2.action_outcomes
    WHERE playbook_id = p_playbook_id;

    UPDATE harvester_v2.playbooks
    SET
        execution_count = v_total,
        success_count = v_success,
        success_rate = CASE WHEN v_total > 0 THEN v_success::FLOAT / v_total ELSE 0 END,
        avg_reward = COALESCE(v_avg_reward, 0),
        updated_at = NOW()
    WHERE playbook_id = p_playbook_id;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate sub-agent metrics
CREATE OR REPLACE FUNCTION harvester_v2.aggregate_subagent_metrics(
    p_subagent_name VARCHAR,
    p_tenant_id VARCHAR,
    p_period_start TIMESTAMPTZ,
    p_period_end TIMESTAMPTZ,
    p_period_type VARCHAR
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO harvester_v2.subagent_metrics (
        subagent_name,
        period_start,
        period_end,
        period_type,
        execution_count,
        success_count,
        failure_count,
        avg_duration_ms,
        p50_duration_ms,
        p95_duration_ms,
        p99_duration_ms,
        total_tokens,
        avg_tokens_per_execution,
        avg_critique_score,
        avg_iterations,
        tenant_id
    )
    SELECT
        p_subagent_name,
        p_period_start,
        p_period_end,
        p_period_type,
        COUNT(*),
        COUNT(*) FILTER (WHERE error_message IS NULL),
        COUNT(*) FILTER (WHERE error_message IS NOT NULL),
        AVG(duration_ms),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms),
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms),
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms),
        SUM(token_count),
        AVG(token_count),
        AVG(critique_score),
        AVG(iteration_count),
        p_tenant_id
    FROM harvester_v2.execution_traces
    WHERE subagent_name = p_subagent_name
      AND tenant_id = p_tenant_id
      AND created_at >= p_period_start
      AND created_at < p_period_end
    ON CONFLICT (subagent_name, period_start, period_type, tenant_id)
    DO UPDATE SET
        execution_count = EXCLUDED.execution_count,
        success_count = EXCLUDED.success_count,
        failure_count = EXCLUDED.failure_count,
        avg_duration_ms = EXCLUDED.avg_duration_ms,
        p50_duration_ms = EXCLUDED.p50_duration_ms,
        p95_duration_ms = EXCLUDED.p95_duration_ms,
        p99_duration_ms = EXCLUDED.p99_duration_ms,
        total_tokens = EXCLUDED.total_tokens,
        avg_tokens_per_execution = EXCLUDED.avg_tokens_per_execution,
        avg_critique_score = EXCLUDED.avg_critique_score,
        avg_iterations = EXCLUDED.avg_iterations;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Trigger to update playbook success rate on outcome insert
CREATE OR REPLACE FUNCTION harvester_v2.on_outcome_insert()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.playbook_id IS NOT NULL THEN
        PERFORM harvester_v2.update_playbook_success_rate(NEW.playbook_id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_outcome_insert
    AFTER INSERT ON harvester_v2.action_outcomes
    FOR EACH ROW
    EXECUTE FUNCTION harvester_v2.on_outcome_insert();

-- ============================================================================
-- Views
-- ============================================================================

-- View: Recent action outcomes with playbook info
CREATE OR REPLACE VIEW harvester_v2.v_recent_outcomes AS
SELECT
    o.id,
    o.action_id,
    o.nba_type,
    o.customer_id,
    o.outcome,
    o.reward,
    o.actual_value,
    o.predicted_value,
    p.playbook_name,
    p.success_rate AS playbook_success_rate,
    o.tenant_id,
    o.created_at
FROM harvester_v2.action_outcomes o
LEFT JOIN harvester_v2.playbooks p ON o.playbook_id = p.playbook_id
ORDER BY o.created_at DESC;

-- View: Sub-agent performance summary (last 24h)
CREATE OR REPLACE VIEW harvester_v2.v_subagent_performance_24h AS
SELECT
    subagent_name,
    tenant_id,
    COUNT(*) AS executions,
    COUNT(*) FILTER (WHERE error_message IS NULL) AS successes,
    ROUND(AVG(duration_ms)::numeric, 2) AS avg_duration_ms,
    ROUND(AVG(critique_score)::numeric, 3) AS avg_critique,
    SUM(token_count) AS total_tokens
FROM harvester_v2.execution_traces
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY subagent_name, tenant_id
ORDER BY executions DESC;

-- View: Training data ready for export
CREATE OR REPLACE VIEW harvester_v2.v_training_data_ready AS
SELECT
    id,
    data_type,
    prompt,
    chosen_response,
    rejected_response,
    chosen_reward,
    rejected_reward,
    quality_score,
    task_type,
    tenant_id,
    created_at
FROM harvester_v2.training_data
WHERE used_for_training = FALSE
  AND quality_score >= 0.7
ORDER BY quality_score DESC, created_at ASC;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON SCHEMA harvester_v2 IS 'Harvester V2 learning and observability tables';
COMMENT ON TABLE harvester_v2.episodic_memories IS 'Cross-session memories (backup from Weaviate)';
COMMENT ON TABLE harvester_v2.playbooks IS 'Success playbooks for procedural memory';
COMMENT ON TABLE harvester_v2.action_outcomes IS 'Action outcomes for reinforcement learning';
COMMENT ON TABLE harvester_v2.training_data IS 'DPO pairs and SFT data for fine-tuning';
COMMENT ON TABLE harvester_v2.execution_traces IS 'Execution traces for observability';
COMMENT ON TABLE harvester_v2.subagent_metrics IS 'Aggregated sub-agent performance metrics';
COMMENT ON TABLE harvester_v2.customer_intelligence_history IS 'Historical customer intelligence snapshots';
