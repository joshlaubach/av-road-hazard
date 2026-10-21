Purpose: ADR 0001: Qdrant as the vector store.

# ADR 0001: Qdrant as the vector store

Status: accepted (owner approved)

## Context
The RAG layer needs metadata filtering plus vector search over image and text embeddings, running locally in Docker and hosted on a free tier.

## Decision
Use Qdrant: filters and multi-vector support, runs in Docker, free cloud tier (Qdrant Cloud). Local: Docker container. Hosted: Qdrant Cloud free tier. <!-- F:R-01 -->

## Alternatives considered
- Chroma: simplest, embedded, weaker filtering. Rejected.
- pgvector: one Postgres for vectors and metadata. Rejected.

## Consequences
One collection holds an image vector, a text vector, and a metadata payload per frame. Adds a Qdrant container to compose and a cloud dependency for the hosted demo. Needs `QDRANT_URL` and `QDRANT_API_KEY`.
