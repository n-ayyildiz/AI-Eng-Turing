-- Neurohypothesis — Supabase table setup
-- ─────────────────────────────────────────────────────────────────────────
-- HOW TO RUN:
--   Supabase dashboard → SQL Editor (left sidebar) → New query
--   → paste this entire file → click Run
-- ─────────────────────────────────────────────────────────────────────────

create table if not exists neurohypothesis_sessions (
    id            bigserial    primary key,
    session_id    text         not null,
    user_id       text         not null,
    topic         text,
    n_hypotheses  integer      default 0,
    avg_rating    numeric(3,2),
    ratings       jsonb,
    cost_usd      numeric(10,6),
    path_choice   text,
    completed_at  timestamptz  default now()
);

-- Row Level Security:
--   anyone can INSERT (your app writes with service_role key)
--   nobody can SELECT via API (only you, via the dashboard or service_role key)
alter table neurohypothesis_sessions enable row level security;

create policy "app_can_insert"
    on neurohypothesis_sessions
    for insert
    with check (true);

-- ─────────────────────────────────────────────────────────────────────────
-- To view your data:
--   Supabase dashboard → Table Editor → neurohypothesis_sessions
-- ─────────────────────────────────────────────────────────────────────────
