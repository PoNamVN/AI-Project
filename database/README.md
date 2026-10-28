# Vehicle database

schema.sql is a draft design, not a populated or integrated database. Vehicles represent dataset categories; variants describe market/trim/powertrain differences. Specifications store provenance per field.

Future specification fields include body type, engine, power, transmission, fuel type, seats, dimensions and consumption when verified. Establish field names/units and validation after categories/sources are approved. Optional pricing is excluded initially. Missing fields stay unknown. Category/variant descriptors also need sourced verification before population; the draft will be refined to track their provenance as needed.

Enable foreign_keys for every SQLite connection. Use ISO verification dates, explicit units and source applicability. class_mapping.json must reference vehicles.category_id. Do not populate fake example vehicles.
