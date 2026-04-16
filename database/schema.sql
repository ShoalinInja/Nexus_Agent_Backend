-- =============================================================================
-- UniAcco AI Sales Intelligence System — Supabase PostgreSQL Schema
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Extensions
-- -----------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;


-- -----------------------------------------------------------------------------
-- Table: managers
-- Booking managers / partners who own properties
-- -----------------------------------------------------------------------------

CREATE TABLE managers (
    manager_id          TEXT PRIMARY KEY,
    manager_name        TEXT NOT NULL,
    booking_process     TEXT,
    eligibility_criteria TEXT,
    priority            INT DEFAULT 0
);


-- -----------------------------------------------------------------------------
-- Table: properties
-- Core property listings with vector embedding for semantic search
-- -----------------------------------------------------------------------------

CREATE TABLE properties (
    property_id             TEXT PRIMARY KEY,
    manager_id              TEXT REFERENCES managers(manager_id),
    property_name           TEXT NOT NULL,
    city                    TEXT NOT NULL,
    description             TEXT,
    commission_pct          DECIMAL(4,2),
    is_soldout              BOOLEAN DEFAULT FALSE,
    amenities               TEXT,
    recon_confirmation_rate DECIMAL(4,3),
    min_rent_pw             NUMERIC,
    available_room_types    TEXT[],
    available_moveins       TEXT[],
    embedding_summary       TEXT,
    embedding               vector(1536),
    content_hash            TEXT,
    last_synced_at          TIMESTAMPTZ DEFAULT now()
);


-- -----------------------------------------------------------------------------
-- Table: property_configs
-- Room-level pricing and availability configurations per property
-- -----------------------------------------------------------------------------

CREATE TABLE property_configs (
    config_id   SERIAL PRIMARY KEY,
    property_id TEXT REFERENCES properties(property_id),
    room_type   TEXT NOT NULL,
    move_in     TEXT NOT NULL,
    lease_weeks INT,
    rent_pw     NUMERIC NOT NULL,
    is_soldout  BOOLEAN DEFAULT FALSE
);


-- -----------------------------------------------------------------------------
-- Table: property_university_map
-- Walking/travel distances from properties to nearby universities
-- -----------------------------------------------------------------------------

CREATE TABLE property_university_map (
    property_id     TEXT REFERENCES properties(property_id),
    university_name TEXT NOT NULL,
    distance_mins   INT,
    distance_km     DECIMAL(4,2),
    PRIMARY KEY (property_id, university_name)
);


-- -----------------------------------------------------------------------------
-- Table: universities
-- Canonical university names with alias list for fuzzy matching
-- -----------------------------------------------------------------------------

CREATE TABLE universities (
    university_id  SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    city           TEXT NOT NULL,
    aliases        TEXT[]
);


-- -----------------------------------------------------------------------------
-- Table: scoring_config
-- Configurable weights for the property scoring engine
-- -----------------------------------------------------------------------------

CREATE TABLE scoring_config (
    factor      TEXT PRIMARY KEY,
    weight      NUMERIC NOT NULL,
    updated_by  TEXT,
    updated_at  TIMESTAMPTZ DEFAULT now()
);


-- -----------------------------------------------------------------------------
-- Table: query_logs
-- Logs every user query with intent, entities, results, and conversion outcome
-- -----------------------------------------------------------------------------

CREATE TABLE query_logs (
    query_id          SERIAL PRIMARY KEY,
    session_id        TEXT,
    agent_name        TEXT,
    raw_query         TEXT,
    detected_intent   TEXT,
    intent_confidence DECIMAL(3,2),
    extracted_entities JSONB,
    result_count      INT,
    properties_shown  JSONB,
    scores            JSONB,
    response_time_ms  INT,
    converted         BOOLEAN DEFAULT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);


-- -----------------------------------------------------------------------------
-- Table: conversation_logs
-- Full conversation history per session/lead for agent context
-- -----------------------------------------------------------------------------

CREATE TABLE conversation_logs (
    session_id  TEXT,
    lead_id     TEXT,
    agent_name  TEXT,
    messages    JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);


-- -----------------------------------------------------------------------------
-- Table: templates
-- Message/response templates for agent-generated outreach
-- -----------------------------------------------------------------------------

CREATE TABLE templates (
    template_id   SERIAL PRIMARY KEY,
    template_type TEXT,
    scenario      TEXT,
    variables     TEXT[],
    body_text     TEXT
);


-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX idx_pc_property ON property_configs(property_id);
CREATE INDEX idx_pc_room     ON property_configs(room_type);
CREATE INDEX idx_pc_rent     ON property_configs(rent_pw);
CREATE INDEX idx_pc_movein   ON property_configs(move_in);
CREATE INDEX idx_pc_soldout  ON property_configs(is_soldout);
CREATE INDEX idx_props_city  ON properties(city);
CREATE INDEX idx_uni_map     ON property_university_map(university_name);

-- HNSW index for fast approximate nearest-neighbour cosine similarity search
CREATE INDEX idx_embedding ON properties
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- -----------------------------------------------------------------------------
-- Materialized view: property_llm_context
-- Pre-joined, LLM-ready context for available properties.
-- Aggregates room configs and nearby universities as JSON arrays.
-- REFRESH MATERIALIZED VIEW property_llm_context; (run after Airflow sync)
-- -----------------------------------------------------------------------------

CREATE MATERIALIZED VIEW property_llm_context AS
SELECT
    p.property_id,
    p.property_name,
    p.city,
    p.description,
    p.amenities,
    p.commission_pct,
    p.recon_confirmation_rate,
    p.min_rent_pw,
    p.available_room_types,
    p.available_moveins,
    p.embedding_summary,
    p.embedding,
    m.manager_id,
    m.manager_name,
    m.booking_process,
    m.eligibility_criteria,
    m.priority,
    COALESCE(
        json_agg(
            DISTINCT jsonb_build_object(
                'config_id',   pc.config_id,
                'room_type',   pc.room_type,
                'move_in',     pc.move_in,
                'lease_weeks', pc.lease_weeks,
                'rent_pw',     pc.rent_pw,
                'is_soldout',  pc.is_soldout
            )
        ) FILTER (WHERE pc.config_id IS NOT NULL),
        '[]'
    ) AS room_configs,
    COALESCE(
        json_agg(
            DISTINCT jsonb_build_object(
                'university_name', pum.university_name,
                'distance_mins',   pum.distance_mins,
                'distance_km',     pum.distance_km
            )
        ) FILTER (WHERE pum.university_name IS NOT NULL),
        '[]'
    ) AS nearby_universities
FROM properties p
LEFT JOIN managers m
    ON p.manager_id = m.manager_id
LEFT JOIN property_configs pc
    ON p.property_id = pc.property_id
LEFT JOIN property_university_map pum
    ON p.property_id = pum.property_id
WHERE p.is_soldout = FALSE
GROUP BY
    p.property_id,
    p.property_name,
    p.city,
    p.description,
    p.amenities,
    p.commission_pct,
    p.recon_confirmation_rate,
    p.min_rent_pw,
    p.available_room_types,
    p.available_moveins,
    p.embedding_summary,
    p.embedding,
    m.manager_id,
    m.manager_name,
    m.booking_process,
    m.eligibility_criteria,
    m.priority;


-- ─── conversations table ─────────────────────────────────────────────────────
-- Replaces property_enquiry_sessions.
-- Stores full conversation state: messages, search filters, soft-delete flag.

CREATE TABLE IF NOT EXISTS conversations (
  conversation_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          TEXT NOT NULL,
  email            TEXT DEFAULT '',
  filters          JSONB DEFAULT '{}',
  messages         JSONB DEFAULT '[]',
  is_deleted       BOOLEAN DEFAULT FALSE,
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id   ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_is_deleted ON conversations(is_deleted);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
