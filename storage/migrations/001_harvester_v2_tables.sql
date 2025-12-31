-- ============================================================================
-- Migration 001: Harvester V2 Tables
-- ============================================================================
-- Run this migration to create all Harvester V2 tables
--
-- Usage:
--   psql -h localhost -U postgres -d nexgai -f 001_harvester_v2_tables.sql
--
-- Or via Python:
--   from shared.database.postgresql_adapter import get_async_session
--   async with get_async_session() as session:
--       await session.execute(text(open('001_harvester_v2_tables.sql').read()))
-- ============================================================================

-- Include the main schema
\i ../postgresql_schema.sql

-- Mark migration as complete
INSERT INTO public.schema_migrations (version, name, applied_at)
VALUES ('001', 'harvester_v2_tables', NOW())
ON CONFLICT (version) DO NOTHING;
