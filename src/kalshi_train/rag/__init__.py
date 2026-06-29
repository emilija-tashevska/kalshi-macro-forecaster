"""Retrieval-augmented generation (RAG) over the Fed text corpus.

``chunk`` splits documents into passages; ``embedder`` turns text into
vectors (OpenAI, lazily imported); ``store`` persists chunk embeddings in
``text_chunks`` and does **point-in-time** retrieval (only passages
published on/before the as-of date). The SFT builder (Phase 4) uses
``store.retrieve`` to fold the most relevant Fed passages into each
training prompt, so the model learns from the Fed's actual words.
"""
