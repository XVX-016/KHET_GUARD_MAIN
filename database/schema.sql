-- Khet Guard Database Schema
-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1️⃣ Users Table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firebase_uid VARCHAR(128) UNIQUE, -- Firebase Auth UID
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    region VARCHAR(100), -- State/District for analytics
    language VARCHAR(10) DEFAULT 'en', -- 'en' or 'hi'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2️⃣ Scans Table
CREATE TABLE IF NOT EXISTS scans (
    scan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,              
    prediction_type VARCHAR(50) NOT NULL, -- 'disease_pest' | 'cattle'
    class_name VARCHAR(100),              
    confidence NUMERIC(5,4),              
    class_id INT,
    grad_cam_url TEXT,                    
    pesticide_recommendation JSONB, -- Store full recommendation object
    breed_info JSONB, -- Store full breed info object
    metadata JSONB, -- Additional scan metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3️⃣ Pesticide Recommendations Table
CREATE TABLE IF NOT EXISTS pesticide_recommendations (
    disease_name VARCHAR(100) PRIMARY KEY,
    recommended TEXT[],                   
    dosage TEXT,
    safety TEXT,
    organic_alternatives TEXT[], -- Organic treatment options
    created_at TIMESTAMP DEFAULT NOW()
);

-- 4️⃣ Cattle Breeds Info Table
CREATE TABLE IF NOT EXISTS cattle_breeds (
    breed_name VARCHAR(100) PRIMARY KEY,
    origin TEXT,
    milk_yield TEXT,
    characteristics TEXT,
    feeding_requirements TEXT,
    vaccination_schedule TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5️⃣ Analytics Table
CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    region VARCHAR(100),
    disease_name VARCHAR(100),
    breed_name VARCHAR(100),
    scan_count INT DEFAULT 0,
    avg_confidence NUMERIC(5,4),
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(region, disease_name, breed_name)
);

-- 6️⃣ User Sessions Table (for tracking usage)
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    device_info JSONB,
    app_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW()
);

-- 7️⃣ Feedback Table (for model improvement)
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES scans(scan_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    is_correct BOOLEAN,
    actual_class VARCHAR(100), -- What the user says it actually is
    comments TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
CREATE INDEX IF NOT EXISTS idx_scans_prediction_type ON scans(prediction_type);
CREATE INDEX IF NOT EXISTS idx_analytics_region ON analytics(region);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_scan_id ON feedback(scan_id);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
