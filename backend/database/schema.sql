-- Users table
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);

-- Translations table
CREATE TABLE translations (
    translation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    full_text TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- ASL Signs table
CREATE TABLE asl_signs (
    sign_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign_text TEXT NOT NULL,
    sign_image BLOB
);

-- Junction table for translations and ASL signs
CREATE TABLE translation_signs (
    translation_id INTEGER,
    sign_id INTEGER,
    FOREIGN KEY (translation_id) REFERENCES translations (translation_id),
    FOREIGN KEY (sign_id) REFERENCES asl_signs (sign_id),
    PRIMARY KEY (translation_id, sign_id)
);