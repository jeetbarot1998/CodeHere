-- Projects table (like Claude's artifact context)
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    team_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table (individual messages for better context search)
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    project_name VARCHAR(255) REFERENCES projects(project_name),
    user_id VARCHAR(255) REFERENCES users(user_id),
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    parent_message_id UUID, -- Links question to answer
    model_used VARCHAR(100),
    starred BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vector_id VARCHAR(255) -- Reference to Qdrant vector ID
);

-- Artifacts table for storing file contents and training data
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    project_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,

    -- File information
    file_path VARCHAR(500),
    file_name VARCHAR(255),
    language VARCHAR(50) NOT NULL,
    file_extension VARCHAR(10),

    -- Content and metadata
    content TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    artifact_type VARCHAR(100) DEFAULT 'code_file',

    -- Processing metadata
    extraction_method VARCHAR(50), -- 'code_block', 'file_mention', 'auto_detected'
    was_large_file BOOLEAN DEFAULT FALSE,
    original_format VARCHAR(50), -- 'markdown_code_block', 'raw_text', etc.

    -- Training metadata
    complexity_score INTEGER DEFAULT 0,
    has_functions BOOLEAN DEFAULT FALSE,
    has_classes BOOLEAN DEFAULT FALSE,
    has_imports BOOLEAN DEFAULT FALSE,
    is_complete_file BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX idx_artifacts_message_id ON artifacts(message_id);
CREATE INDEX idx_artifacts_project_session ON artifacts(project_name, session_id);
CREATE INDEX idx_artifacts_language ON artifacts(language);
CREATE INDEX idx_artifacts_file_path ON artifacts(file_path);
CREATE INDEX idx_artifacts_user_id ON artifacts(user_id);
CREATE INDEX idx_artifacts_created_at ON artifacts(created_at);
CREATE INDEX idx_artifacts_size ON artifacts(size_bytes);


-- Create indexes for better performance
CREATE INDEX idx_messages_project_name ON messages(project_name);
CREATE INDEX idx_messages_user_id ON messages(user_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_starred ON messages(starred);
CREATE INDEX idx_messages_parent_id ON messages(parent_message_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Update timestamp trigger for artifacts
CREATE TRIGGER update_artifacts_updated_at
    BEFORE UPDATE ON artifacts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();