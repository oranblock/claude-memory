-- recommendation_system_schema.sql
-- Database schema for the e-commerce recommendation system

-- Products table
CREATE TABLE products (
    product_id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category_id VARCHAR(36) NOT NULL,
    subcategory_id VARCHAR(36),
    brand_id VARCHAR(36),
    average_rating DECIMAL(3, 2),
    review_count INTEGER DEFAULT 0,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT fk_category FOREIGN KEY (category_id) REFERENCES categories(category_id),
    CONSTRAINT fk_subcategory FOREIGN KEY (subcategory_id) REFERENCES subcategories(subcategory_id),
    CONSTRAINT fk_brand FOREIGN KEY (brand_id) REFERENCES brands(brand_id)
);

-- Product attributes for recommendation features
CREATE TABLE product_attributes (
    product_id VARCHAR(36) NOT NULL,
    attribute_id VARCHAR(36) NOT NULL,
    attribute_value VARCHAR(255),
    attribute_numeric_value DECIMAL(10, 2),
    PRIMARY KEY (product_id, attribute_id),
    CONSTRAINT fk_product_attr FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT fk_attribute FOREIGN KEY (attribute_id) REFERENCES attributes(attribute_id)
);

-- User table
CREATE TABLE users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    registration_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- User preferences for content-based recommendations
CREATE TABLE user_preferences (
    user_id VARCHAR(36) NOT NULL,
    category_id VARCHAR(36) NOT NULL,
    preference_score DECIMAL(4, 3) NOT NULL,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, category_id),
    CONSTRAINT fk_user_pref FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_category_pref FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- User interactions for collaborative filtering
CREATE TABLE user_interactions (
    interaction_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    interaction_type ENUM('view', 'add_to_cart', 'purchase', 'wishlist', 'review') NOT NULL,
    interaction_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(36),
    reference_id VARCHAR(36), -- For order_id or review_id
    interaction_value INTEGER, -- For ratings or counts
    CONSTRAINT fk_user_int FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_product_int FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- User segments for targeted recommendations
CREATE TABLE user_segments (
    segment_id VARCHAR(36) PRIMARY KEY,
    segment_name VARCHAR(100) NOT NULL,
    description TEXT,
    creation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    segment_rules JSON, -- Rules for dynamic segments
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- User segment memberships
CREATE TABLE user_segment_members (
    segment_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    added_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (segment_id, user_id),
    CONSTRAINT fk_segment FOREIGN KEY (segment_id) REFERENCES user_segments(segment_id) ON DELETE CASCADE,
    CONSTRAINT fk_user_segment FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Recommendation history for analysis and improvement
CREATE TABLE recommendation_history (
    recommendation_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    recommendation_algorithm VARCHAR(50) NOT NULL,
    recommendation_score DECIMAL(5, 4),
    position_shown INTEGER,
    recommendation_context VARCHAR(50), -- e.g., 'product_page', 'homepage', 'email'
    was_clicked BOOLEAN,
    was_purchased BOOLEAN,
    recommendation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_rec FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_product_rec FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Similar products for content-based recommendations
CREATE TABLE similar_products (
    product_id VARCHAR(36) NOT NULL,
    similar_product_id VARCHAR(36) NOT NULL,
    similarity_score DECIMAL(5, 4) NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id, similar_product_id),
    CONSTRAINT fk_product_sim FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT fk_similar_product FOREIGN KEY (similar_product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

-- A/B testing for recommendation algorithms
CREATE TABLE recommendation_experiments (
    experiment_id VARCHAR(36) PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    allocation_percentage INTEGER NOT NULL DEFAULT 50, -- Percentage of users in test group
    control_algorithm VARCHAR(50) NOT NULL,
    test_algorithm VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_user_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_user_interactions_product ON user_interactions(product_id);
CREATE INDEX idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX idx_product_category ON products(category_id);
CREATE INDEX idx_similar_products_score ON similar_products(similarity_score);
CREATE INDEX idx_recommendation_history_user ON recommendation_history(user_id);
CREATE INDEX idx_recommendation_history_algorithm ON recommendation_history(recommendation_algorithm);