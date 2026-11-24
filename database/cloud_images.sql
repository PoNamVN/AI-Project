-- Image bytes belong in Storage. This catalog is server-only until team auth is configured.
create table public.carvision_images (
    dataset text not null,
    relative_path text not null,
    class_id text not null check (class_id in ('model_s','model_3','model_x','model_y','cybertruck')),
    purpose text not null check (purpose in ('baseline','evaluation_only','supplement_pending')),
    decision text not null check (decision in ('keep','exclude')),
    review_status text not null,
    split text not null check (split in ('train','validation','test','unsplit')),
    sha256 text not null check (sha256 ~ '^[a-f0-9]{64}$'),
    object_path text not null check (object_path ~ '^sha256/[a-f0-9]{64}\.(jpg|jpeg|png|webp|bmp|tif|tiff)$'),
    byte_size bigint not null check (byte_size > 0 and byte_size <= 30000000),
    source_url text not null,
    license text not null,
    provenance jsonb not null,
    uploaded boolean not null default false,
    created_at timestamptz not null default now(),
    primary key (dataset, relative_path)
);
alter table public.carvision_images enable row level security;
revoke all on public.carvision_images from anon, authenticated;
grant select, insert, update on public.carvision_images to service_role;
create index carvision_images_sha256_idx on public.carvision_images (sha256);
