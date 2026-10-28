-- Draft only. No factual vehicle records are supplied.
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS vehicles (
    category_id TEXT PRIMARY KEY,
    dataset_label TEXT NOT NULL UNIQUE,
    manufacturer TEXT NOT NULL,
    model TEXT NOT NULL,
    model_year INTEGER CHECK (model_year BETWEEN 1886 AND 2200)
);
CREATE TABLE IF NOT EXISTS variants (
    variant_id INTEGER PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES vehicles(category_id),
    market TEXT NOT NULL,
    variant_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    reference TEXT NOT NULL,
    publisher TEXT,
    applicability_notes TEXT
);
CREATE TABLE IF NOT EXISTS specifications (
    variant_id INTEGER NOT NULL REFERENCES variants(variant_id),
    field_name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    unit TEXT,
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    verified_on TEXT NOT NULL,
    PRIMARY KEY (variant_id, field_name)
);
