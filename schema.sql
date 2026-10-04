-- ============================================================
-- Schéma Supabase pour la plateforme de recherche de normes LPEE
-- À exécuter UNE FOIS dans Supabase : Dashboard > SQL Editor > New query
--
-- ⚠️ MISE À JOUR (v2) : si vous avez déjà exécuté une version précédente de ce
-- script, la table "documents" existante n'a pas les bonnes colonnes et TOUS
-- les enregistrements ont échoué (erreur PGRST100) — elle est donc vide.
-- Avant de relancer ce script, supprimez-la d'abord :
--     drop table if exists documents;
-- ============================================================

-- 1) Activer l'extension vectorielle (une seule fois par projet Supabase)
create extension if not exists vector;

-- 2) Table qui contient chaque "chunk" (morceau) de document, avec son embedding
create table if not exists documents (
    id          bigserial primary key,
    fichier     text not null,               -- nom du fichier PDF (colonne réelle, pour la contrainte unique)
    chunk_id    text not null,                -- identifiant du chunk au sein du fichier (ex: "12-3")
    content     text not null,               -- le texte du chunk
    metadata    jsonb not null default '{}',  -- {"fichier": "...", "page": 12, "chemin": "..."}
    embedding   vector(384),                  -- vecteur du modèle local multilingual-e5-small (gratuit, illimité)
    created_at  timestamptz not null default now(),
    unique (fichier, chunk_id)                -- nécessaire pour que upsert(..., on_conflict="fichier,chunk_id") fonctionne :
                                               -- PostgREST n'accepte on_conflict que sur de vraies colonnes/contraintes,
                                               -- pas sur des expressions comme metadata->>'fichier'.
);

-- 3) Index de recherche approximative par similarité (rapide même sur des dizaines de milliers de chunks)
--    À exécuter APRÈS avoir ingéré au moins quelques centaines de lignes pour de meilleures performances.
create index if not exists documents_embedding_idx
    on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- 4) Fonction appelée par l'application pour trouver les chunks les plus proches d'une question
create or replace function match_documents (
    query_embedding vector(384),
    match_count int default 6
) returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;
