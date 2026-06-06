-- Neurohypothesis — add JSONB detail columns to sessions table
-- Run in Supabase SQL Editor (safe to run even if columns already exist)

ALTER TABLE neurohypothesis_sessions
    ADD COLUMN IF NOT EXISTS hypotheses_summary  JSONB,
    ADD COLUMN IF NOT EXISTS hypotheses_detail   JSONB,
    ADD COLUMN IF NOT EXISTS token_usage_detail  JSONB,
    ADD COLUMN IF NOT EXISTS errors_detail       JSONB;

-- Ensure service_role has full access (run this too if needed)
GRANT ALL ON ALL TABLES    IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
