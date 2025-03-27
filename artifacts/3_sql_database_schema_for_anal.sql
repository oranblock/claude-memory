-- analytics_platform_schema.sql
-- Comprehensive schema for a data analytics platform

-- Users and Authentication
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    account_status VARCHAR(20) NOT NULL DEFAULT 'active',
    role VARCHAR(20) NOT NULL DEFAULT 'analyst',
    CONSTRAINT chk_account_status CHECK (account_status IN ('active', 'inactive', 'suspended', 'deleted')),
    CONSTRAINT chk_role CHECK (role IN ('admin', 'manager', 'analyst', 'viewer', 'guest'))
);

-- User profiles with additional information
CREATE TABLE user_profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    company VARCHAR(100),
    job_title VARCHAR(100),
    department VARCHAR(100),
    phone VARCHAR(30),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language_preference VARCHAR(10) DEFAULT 'en',
    theme_preference VARCHAR(20) DEFAULT 'light',
    notification_settings JSONB DEFAULT '{"email": true, "in_app": true, "reports": true}',
    bio TEXT,
    avatar_url VARCHAR(255),
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_user_profile UNIQUE (user_id)
);

-- Teams and Organizations
CREATE TABLE organizations (
    org_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    domain VARCHAR(100),
    plan_type VARCHAR(50) NOT NULL DEFAULT 'standard',
    max_users INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    billing_email VARCHAR(100),
    contact_name VARCHAR(100),
    contact_phone VARCHAR(30),
    settings JSONB,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    settings JSONB,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE team_members (
    team_id INTEGER NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'member',
    joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    invited_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (team_id, user_id),
    CONSTRAINT chk_team_role CHECK (role IN ('admin', 'owner', 'member'))
);

-- Data Sources and Connections
CREATE TABLE data_sources (
    source_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    connection_details JSONB NOT NULL,
    credentials_secret_id VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    enabled BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_source_type CHECK (type IN 
        ('database', 'api', 'file_upload', 's3', 'gcs', 'azure_blob', 
         'bigquery', 'snowflake', 'redshift', 'google_sheets', 'excel'))
);

CREATE TABLE data_source_permissions (
    source_id INTEGER NOT NULL REFERENCES data_sources(source_id) ON DELETE CASCADE,
    entity_type VARCHAR(10) NOT NULL,
    entity_id INTEGER NOT NULL,
    permission_level VARCHAR(20) NOT NULL,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (source_id, entity_type, entity_id),
    CONSTRAINT chk_entity_type CHECK (entity_type IN ('user', 'team')),
    CONSTRAINT chk_permission_level CHECK (permission_level IN 
        ('owner', 'editor', 'viewer', 'uploader'))
);

-- Dataset Management
CREATE TABLE datasets (
    dataset_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    source_id INTEGER REFERENCES data_sources(source_id),
    source_query TEXT,
    schema_definition JSONB,
    row_count INTEGER,
    column_count INTEGER,
    file_size_bytes BIGINT,
    file_format VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_updated_at TIMESTAMP,
    last_synced_at TIMESTAMP,
    refresh_schedule VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    tags TEXT[],
    CONSTRAINT chk_file_format CHECK (file_format IN 
        ('csv', 'json', 'parquet', 'avro', 'excel', 'sql', 'unknown'))
);

CREATE TABLE dataset_versions (
    version_id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    storage_location VARCHAR(255) NOT NULL,
    change_description TEXT,
    row_count INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    is_current BOOLEAN DEFAULT FALSE,
    CONSTRAINT unique_dataset_version UNIQUE (dataset_id, version_number)
);

CREATE TABLE dataset_columns (
    column_id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    column_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(100),
    description TEXT,
    data_type VARCHAR(50) NOT NULL,
    is_nullable BOOLEAN DEFAULT TRUE,
    is_unique BOOLEAN DEFAULT FALSE,
    is_indexed BOOLEAN DEFAULT FALSE,
    is_primary_key BOOLEAN DEFAULT FALSE,
    is_foreign_key BOOLEAN DEFAULT FALSE,
    referenced_table VARCHAR(100),
    referenced_column VARCHAR(100),
    position INTEGER NOT NULL,
    example_values TEXT[],
    statistics JSONB,
    CONSTRAINT unique_column_position UNIQUE (dataset_id, position),
    CONSTRAINT unique_column_name UNIQUE (dataset_id, column_name)
);

-- Analysis and Dashboards
CREATE TABLE dashboards (
    dashboard_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    layout JSONB,
    theme VARCHAR(20) DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    is_published BOOLEAN DEFAULT FALSE,
    published_at TIMESTAMP,
    published_by INTEGER REFERENCES users(user_id),
    view_count INTEGER DEFAULT 0,
    tags TEXT[],
    thumbnail_url VARCHAR(255)
);

CREATE TABLE dashboard_permissions (
    dashboard_id INTEGER NOT NULL REFERENCES dashboards(dashboard_id) ON DELETE CASCADE,
    entity_type VARCHAR(10) NOT NULL,
    entity_id INTEGER NOT NULL,
    permission_level VARCHAR(20) NOT NULL,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (dashboard_id, entity_type, entity_id),
    CONSTRAINT chk_entity_type CHECK (entity_type IN ('user', 'team')),
    CONSTRAINT chk_permission_level CHECK (permission_level IN 
        ('owner', 'editor', 'viewer'))
);

CREATE TABLE visualizations (
    visualization_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    query_definition JSONB NOT NULL,
    chart_config JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    last_run_at TIMESTAMP,
    run_time_ms INTEGER,
    error_message TEXT,
    thumbnail_url VARCHAR(255),
    CONSTRAINT chk_visualization_type CHECK (type IN 
        ('bar', 'line', 'pie', 'scatter', 'area', 'table', 'pivot', 
         'heatmap', 'map', 'funnel', 'gauge', 'kpi', 'histogram', 'box'))
);

CREATE TABLE dashboard_visualizations (
    dashboard_id INTEGER NOT NULL REFERENCES dashboards(dashboard_id) ON DELETE CASCADE,
    visualization_id INTEGER NOT NULL REFERENCES visualizations(visualization_id) ON DELETE CASCADE,
    position_config JSONB NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (dashboard_id, visualization_id)
);

-- Reports and Schedules
CREATE TABLE reports (
    report_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dashboard_id INTEGER REFERENCES dashboards(dashboard_id),
    format VARCHAR(20) NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    is_public BOOLEAN DEFAULT FALSE,
    public_url_token VARCHAR(100),
    CONSTRAINT chk_report_format CHECK (format IN 
        ('pdf', 'excel', 'csv', 'html', 'json'))
);

CREATE TABLE scheduled_tasks (
    task_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    task_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(20) NOT NULL,
    entity_id INTEGER NOT NULL,
    schedule_expression VARCHAR(100) NOT NULL,
    schedule_timezone VARCHAR(50) DEFAULT 'UTC',
    parameters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    last_status VARCHAR(20),
    last_error TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_task_type CHECK (task_type IN 
        ('report_email', 'dashboard_refresh', 'dataset_sync', 'alert_check')),
    CONSTRAINT chk_entity_type CHECK (entity_type IN 
        ('report', 'dashboard', 'dataset', 'alert'))
);

CREATE TABLE task_recipients (
    task_id INTEGER NOT NULL REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
    recipient_type VARCHAR(10) NOT NULL,
    recipient_value VARCHAR(100) NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (task_id, recipient_type, recipient_value),
    CONSTRAINT chk_recipient_type CHECK (recipient_type IN 
        ('email', 'slack', 'webhook', 'user_id', 'team_id'))
);

-- Alerts and Monitoring
CREATE TABLE alerts (
    alert_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    query_definition JSONB NOT NULL,
    condition_type VARCHAR(20) NOT NULL,
    condition_value NUMERIC,
    comparison_type VARCHAR(20) NOT NULL,
    time_window INTEGER,
    time_window_unit VARCHAR(10),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_check_at TIMESTAMP,
    last_triggered_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    severity VARCHAR(10) DEFAULT 'medium',
    CONSTRAINT chk_condition_type CHECK (condition_type IN 
        ('threshold', 'change', 'anomaly', 'absence')),
    CONSTRAINT chk_comparison_type CHECK (comparison_type IN 
        ('greater_than', 'less_than', 'equal_to', 'not_equal_to', 
         'percentage_increase', 'percentage_decrease')),
    CONSTRAINT chk_time_window_unit CHECK (time_window_unit IN 
        ('minute', 'hour', 'day', 'week')),
    CONSTRAINT chk_severity CHECK (severity IN 
        ('low', 'medium', 'high', 'critical'))
);

CREATE TABLE alert_recipients (
    alert_id INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    recipient_type VARCHAR(10) NOT NULL,
    recipient_value VARCHAR(100) NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (alert_id, recipient_type, recipient_value),
    CONSTRAINT chk_recipient_type CHECK (recipient_type IN 
        ('email', 'slack', 'webhook', 'user_id', 'team_id'))
);

CREATE TABLE alert_history (
    history_id SERIAL PRIMARY KEY,
    alert_id INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual_value NUMERIC,
    comparison_value NUMERIC,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_type VARCHAR(20),
    resolution_note TEXT,
    CONSTRAINT chk_resolution_type CHECK (resolution_type IN 
        ('auto', 'manual', 'acknowledged'))
);

-- Audit and Activity Tracking
CREATE TABLE activity_log (
    log_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id),
    user_id INTEGER REFERENCES users(user_id),
    activity_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INTEGER,
    action VARCHAR(20) NOT NULL,
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_action CHECK (action IN 
        ('create', 'read', 'update', 'delete', 'login', 'logout', 
         'export', 'share', 'run', 'refresh'))
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status_role ON users(account_status, role);
CREATE INDEX idx_team_members_user ON team_members(user_id);
CREATE INDEX idx_datasets_org ON datasets(org_id);
CREATE INDEX idx_datasets_source ON datasets(source_id);
CREATE INDEX idx_dashboards_org ON dashboards(org_id);
CREATE INDEX idx_dashboards_created_by ON dashboards(created_by);
CREATE INDEX idx_visualizations_dataset ON visualizations(dataset_id);
CREATE INDEX idx_activity_log_timestamp ON activity_log(timestamp);
CREATE INDEX idx_activity_log_user ON activity_log(user_id);
CREATE INDEX idx_activity_log_type_action ON activity_log(activity_type, action);

-- Create useful views for common queries
CREATE VIEW active_users_view AS
    SELECT 
        u.user_id, 
        u.username, 
        u.email, 
        u.first_name, 
        u.last_name, 
        u.role,
        p.company,
        p.job_title,
        u.last_login_at,
        u.created_at,
        COUNT(DISTINCT tm.team_id) AS team_count
    FROM users u
    LEFT JOIN user_profiles p ON u.user_id = p.user_id
    LEFT JOIN team_members tm ON u.user_id = tm.user_id
    WHERE u.account_status = 'active'
    GROUP BY u.user_id, p.profile_id;

CREATE VIEW dashboard_analytics_view AS
    SELECT
        d.dashboard_id,
        d.name,
        d.created_at,
        u_created.username AS created_by_user,
        d.last_modified_at,
        u_modified.username AS modified_by_user,
        d.view_count,
        d.is_published,
        COUNT(dv.visualization_id) AS visualization_count,
        MAX(d.last_modified_at) AS last_update
    FROM dashboards d
    LEFT JOIN users u_created ON d.created_by = u_created.user_id
    LEFT JOIN users u_modified ON d.last_modified_by = u_modified.user_id
    LEFT JOIN dashboard_visualizations dv ON d.dashboard_id = dv.dashboard_id
    GROUP BY 
        d.dashboard_id, 
        d.name, 
        d.created_at, 
        u_created.username, 
        d.last_modified_at, 
        u_modified.username,
        d.view_count,
        d.is_published;

CREATE VIEW dataset_usage_view AS
    SELECT
        ds.dataset_id,
        ds.name,
        ds.created_at,
        ds.row_count,
        ds.last_synced_at,
        COUNT(DISTINCT v.visualization_id) AS visualization_count,
        COUNT(DISTINCT dv.dashboard_id) AS dashboard_count,
        COUNT(DISTINCT a.alert_id) AS alert_count
    FROM datasets ds
    LEFT JOIN visualizations v ON ds.dataset_id = v.dataset_id
    LEFT JOIN dashboard_visualizations dv ON v.visualization_id = dv.visualization_id
    LEFT JOIN alerts a ON ds.dataset_id = a.dataset_id
    GROUP BY
        ds.dataset_id,
        ds.name,
        ds.created_at,
        ds.row_count,
        ds.last_synced_at;