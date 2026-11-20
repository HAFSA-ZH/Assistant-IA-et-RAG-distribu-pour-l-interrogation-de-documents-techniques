#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spark_rag_pipeline.py - Mini-Projet 05 : Assistant IA & RAG distribué
=====================================================================

Pipeline RAG distribué pour le corpus EnterpriseRAG-Bench :

    Parquet  --Spark-->  chunking  --Spark-->  embeddings  -->  FAISS + SQLite
                                                                   |
        question --> embedding --> top-k (cosinus) --> prompt --> LLM (Ollama)

Sous-commandes
--------------
    download   Télécharge le dataset depuis Hugging Face
    build      Ingestion + chunking + embeddings (Spark) + indexation (FAISS)
    search     Recherche sémantique en ligne de commande
    ask        Question complète (retrieval + génération Ollama)
    ftsindex   Ajoute l'index BM25 -> recherche hybride (vecteurs + mots-clés)

Exemples
--------
    python spark_rag_pipeline.py download --out data/enterprise_rag
    python spark_rag_pipeline.py build --data data/enterprise_rag --out rag_index --limit-docs 20000
    python spark_rag_pipeline.py search --index rag_index "What is the SLO of the billing API?"
    python spark_rag_pipeline.py ask --index rag_index --model llama3 "Who owns the billing API?"

Le module est aussi importé par app_rag_assistant.py et evaluate_rag.py
(classes Embedder, Retriever, fonction answer_question).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import Dict, Iterable, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Fichier .env (clé API, choix du LLM) : lu automatiquement, sans dépendance externe
# --------------------------------------------------------------------------- #
def load_env_file() -> None:
    """Charge KEY=VALUE depuis un .env (dossier courant ou dossier du script) sans écraser l'environnement."""
    for folder in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        path = os.path.join(folder, ".env")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
            return


load_env_file()


# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
DEFAULT_REPO = "UCG4879/EnterpriseRAG-Bench"
DEFAULT_EMBEDDER = "st:BAAI/bge-small-en-v1.5"   # corpus en anglais
HASH_DIM = 8192   # grande dimension : limite les collisions du hachage
_STOP = set("a an the and or of to in on for with by at from as is are was were be been it its this that "
            "these those which who whom what when where how while under across during".split())
NO_ANSWER = "[[NO_ANSWER]]"

INDEX_FILE = "faiss.index"
SQLITE_FILE = "chunks.sqlite"
META_FILE = "index_meta.json"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """Encodeur de textes -> vecteurs L2-normalisés (cosinus = produit scalaire).

    spec :
      "st:<nom_modèle>"  sentence-transformers (ex. st:BAAI/bge-small-en-v1.5,
                         st:intfloat/multilingual-e5-small pour questions FR)
      "hash"             hachage de n-grammes (sans modèle, déterministe).
                         Utile pour tests hors-ligne / CI uniquement : qualité
                         sémantique très inférieure à un vrai modèle.
    """

    def __init__(self, spec: str = DEFAULT_EMBEDDER):
        self.spec = spec
        self._model = None
        if spec == "hash":
            self.dim = HASH_DIM
        elif spec.startswith("st:"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(spec[3:])
            getdim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self.dim = int(getdim())
        else:
            raise ValueError(f"Embedder inconnu : {spec!r} (utiliser 'hash' ou 'st:<modèle>')")

    # -- préfixes recommandés par certains modèles ---------------------------
    def _prefixes(self):
        name = self.spec.lower()
        if "bge" in name and "m3" not in name:
            return "Represent this sentence for searching relevant passages: ", ""
        if "e5" in name:
            return "query: ", "passage: "
        return "", ""

    @staticmethod
    def _hash_encode(texts: List[str], dim: int) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = re.findall(r"\w+", text.lower())
            feats = [(t, 0.2 if t in _STOP else 1.0) for t in toks]
            feats += [(a + "_" + b, 0.2 if (a in _STOP and b in _STOP) else 1.0)
                      for a, b in zip(toks, toks[1:])]
            for f, w in feats:
                h = zlib.crc32(f.encode("utf-8"))      # stable entre workers
                out[r, h % dim] += w if (h >> 20) & 1 else -w
        out = np.sign(out) * np.log1p(np.abs(out))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)

    def _encode(self, texts: List[str], prefix: str, batch_size: int) -> np.ndarray:
        texts = [prefix + t for t in texts]
        if self.spec == "hash":
            return self._hash_encode(texts, self.dim)
        vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def encode_docs(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self._encode(list(texts), self._prefixes()[1], batch_size)

    def encode_queries(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self._encode(list(texts), self._prefixes()[0], batch_size)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = 150, overlap: int = 30) -> List[str]:
    """Découpe un texte en fenêtres de `size` mots avec `overlap` mots de recouvrement."""
    words = (text or "").split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- #
# Découverte des fichiers du dataset (le schéma exact est auto-détecté)
# --------------------------------------------------------------------------- #
def _parquet_columns(path: str) -> List[str]:
    import pyarrow.parquet as pq
    return list(pq.ParquetFile(path).schema_arrow.names)


def find_document_files(data_dir: str) -> List[str]:
    """Retourne les fichiers parquet/jsonl contenant les documents
    (colonnes doc_id + content, sans colonne 'question')."""
    files = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)):
        try:
            cols = set(_parquet_columns(path))
        except Exception:
            continue
        if {"doc_id", "content"} <= cols and "question" not in cols:
            files.append(path)
    if not files:
        for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)):
            with open(path, "r", encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
            if "doc_id" in first and "content" in first:
                files.append(path)
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier de documents (colonnes doc_id/content) trouvé sous {data_dir}. "
            "Vérifier le dossier ou lancer d'abord la commande 'download'.")
    return files


def load_gold_doc_ids(data_dir: str) -> List[str]:
    """Union des expected_doc_ids de tous les fichiers de questions (lecture de cette colonne seule)."""
    import pyarrow.parquet as pq
    ids = set()
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)):
        try:
            if "expected_doc_ids" in _parquet_columns(path):
                col = pq.read_table(path, columns=["expected_doc_ids"]).column(0).to_pylist()
                ids.update(d for lst in col for d in (lst or []))
        except Exception:
            continue
    return sorted(ids)


# --------------------------------------------------------------------------- #
# Commande : download
# --------------------------------------------------------------------------- #
def cmd_download(args):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=args.out)
    print(f"Dataset téléchargé dans : {path}")


# --------------------------------------------------------------------------- #
# Commande : build  (Spark -> chunks+embeddings -> FAISS + SQLite)
# --------------------------------------------------------------------------- #
def _spark_stage(args, work_dir: str):
    """Étapes Spark : ingestion, chunking, embeddings -> parquet. Retourne (n_docs, t_ingest, t_embed)."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = (SparkSession.builder.master(args.master).appName("EnterpriseRAG-Ingestion")
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", str(args.partitions))
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        spark.sparkContext.addPyFile(os.path.abspath(__file__))   # workers d'un vrai cluster
    except Exception:
        pass

    # ---- 1. Ingestion ------------------------------------------------------
    t0 = time.time()
    files = find_document_files(args.data)
    print(f"[ingestion] {len(files)} fichier(s) de documents détecté(s)")
    reader = spark.read.parquet if files[0].endswith(".parquet") else spark.read.json
    docs = reader(*files)
    for col, default in (("title", ""), ("source_type", "unknown")):
        if col not in docs.columns:
            docs = docs.withColumn(col, F.lit(default))
    docs = docs.select(F.col("doc_id").cast("string").alias("doc_id"),
                       F.coalesce(F.col("source_type").cast("string"), F.lit("unknown")).alias("source_type"),
                       F.coalesce(F.col("title").cast("string"), F.lit("")).alias("title"),
                       F.coalesce(F.col("content").cast("string"), F.lit("")).alias("content"))
    if args.sources:
        wanted = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
        docs = docs.filter(F.lower(F.col("source_type")).isin(wanted))
    if args.limit_docs or args.sample_frac:
        # Échantillon ALÉATOIRE (toutes sources mélangées), pas les N premières lignes du fichier
        total = docs.count()
        frac = args.sample_frac or min(1.0, args.limit_docs * 1.15 / max(total, 1))
        sampled = docs.filter(F.rand(args.seed) < frac) if frac < 1.0 else docs
        if args.limit_docs:
            sampled = sampled.limit(args.limit_docs)
        if args.include_gold:
            gold = load_gold_doc_ids(args.include_gold)
            print(f"[ingestion] {len(gold)} documents attendus par les questions ajoutés à l'échantillon")
            sampled = sampled.unionByName(docs.filter(F.col("doc_id").isin(gold))).dropDuplicates(["doc_id"])
        docs = sampled
    docs = docs.repartition(args.partitions).cache()
    n_docs = docs.count()
    t_ingest = time.time() - t0
    print(f"[ingestion] {n_docs} documents ({t_ingest:.1f}s)")

    # ---- 2. Chunking distribué --------------------------------------------
    size, overlap = args.chunk_size, args.chunk_overlap
    chunk_schema = T.StructType([
        T.StructField("doc_id", T.StringType()), T.StructField("section", T.StringType()),
        T.StructField("chunk_id", T.IntegerType()), T.StructField("source_type", T.StringType()),
        T.StructField("n_chunks", T.IntegerType()), T.StructField("text", T.StringType())])
    chunk_cols = ["doc_id", "section", "chunk_id", "source_type", "n_chunks", "text"]

    def chunk_batches(batches):
        import pandas as pd
        for pdf in batches:
            rows = []
            for d, title, src, content in zip(pdf["doc_id"], pdf["title"], pdf["source_type"], pdf["content"]):
                pieces = chunk_text(content, size, overlap)
                rows.extend((d, title, i, src, len(pieces), p) for i, p in enumerate(pieces))
            if rows:
                yield pd.DataFrame(rows, columns=chunk_cols)

    chunks = docs.mapInPandas(chunk_batches, schema=chunk_schema)

    # ---- 3. Embeddings distribués (un modèle chargé par partition) --------
    emb_schema = T.StructType(chunk_schema.fields + [T.StructField("embedding", T.ArrayType(T.FloatType()))])
    spec = args.embedder

    def embed_batches(batches):
        embedder = Embedder(spec)
        if embedder._model is not None:
            if args.max_seq_length:
                embedder._model.max_seq_length = args.max_seq_length
            import torch
            if args.torch_threads:
                torch.set_num_threads(args.torch_threads)
            if torch.cuda.is_available():          # GPU (ex. T4 Colab) : demi-précision = ~2x plus rapide
                embedder._model.half()
        for pdf in batches:
            texts = [f"{t}\n{x}" if t else x for t, x in zip(pdf["section"], pdf["text"])]
            vecs = embedder.encode_docs(texts, batch_size=args.embed_batch)
            out = pdf.copy()
            out["embedding"] = [v for v in vecs]
            yield out

    t0 = time.time()
    chunks.mapInPandas(embed_batches, schema=emb_schema) \
          .write.mode("overwrite").parquet(work_dir)
    t_embed = time.time() - t0
    docs.unpersist()
    spark.stop()
    print(f"[chunking+embeddings] terminé ({t_embed:.1f}s)")
    return n_docs, t_ingest, t_embed


def cmd_build(args):
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    work_dir = os.path.join(args.out, "_chunks_embeddings")

    size, overlap, spec = args.chunk_size, args.chunk_overlap, args.embedder
    if args.reuse_embeddings:
        # Reprise après un échec d'indexation : on réutilise les embeddings déjà calculés
        import pyarrow.dataset as pads0
        ids = pads0.dataset(work_dir, format="parquet").to_table(columns=["doc_id"]).column(0)
        n_docs = len(set(ids.to_pylist()))
        t_ingest, t_embed = args.prev_ingest_seconds, args.prev_embed_seconds
        print(f"[reprise] embeddings existants réutilisés ({n_docs} documents)")
    else:
        n_docs, t_ingest, t_embed = _spark_stage(args, work_dir)

    # ---- 4. Indexation FAISS + stockage SQLite -----------------------------
    import faiss
    import pyarrow.dataset as pads
    t0 = time.time()
    dataset = pads.dataset(work_dir, format="parquet")
    n_chunks = dataset.count_rows()
    if n_chunks == 0:
        raise RuntimeError("Aucun chunk produit : vérifier les données d'entrée.")
    first = next(dataset.to_batches(batch_size=1, columns=["embedding"]))
    dim = len(first.column(0)[0])

    def make_index(n_total: int):
        if args.index == "flat":
            return faiss.IndexFlatIP(dim)
        if args.index == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 100
            return idx
        nlist = int(min(max(16, 4 * math.sqrt(n_total)), 65536))
        if args.index == "ivfpq":      # vecteurs compressés (m octets/vecteur) : peu de RAM
            return faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, max(1, dim // 8), 8,
                                    faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)

    db_path = os.path.join(args.out, SQLITE_FILE)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id TEXT, section TEXT, chunk_id INTEGER, "
                "n_chunks INTEGER, source_type TEXT, text TEXT)")
    con.execute("CREATE INDEX idx_doc ON chunks(doc_id)")

    index = make_index(n_chunks)
    needs_training = args.index in ("ivf", "ivfpq")
    next_id, buffer_vecs, buffer_rows = 0, [], []

    def flush(vecs, rows):
        nonlocal next_id
        index.add(vecs)
        con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        [(next_id + i, *r) for i, r in enumerate(rows)])
        next_id += len(rows)

    cols = ["embedding", "doc_id", "section", "chunk_id", "n_chunks", "source_type", "text"]
    for batch in dataset.to_batches(batch_size=20000, columns=cols):
        arr = batch.column(0).combine_chunks() if hasattr(batch.column(0), "combine_chunks") else batch.column(0)
        vecs = np.asarray(arr.values, dtype=np.float32).reshape(len(arr), dim)
        rows = list(zip(*(batch.column(i).to_pylist() for i in range(1, 7))))
        if needs_training and not index.is_trained:
            buffer_vecs.append(vecs)
            buffer_rows.extend(rows)
            if sum(len(v) for v in buffer_vecs) >= args.train_size:
                all_v = np.vstack(buffer_vecs)
                index.train(all_v)
                flush(all_v, buffer_rows)
                buffer_vecs, buffer_rows = [], []
        else:
            flush(vecs, rows)
    if needs_training and not index.is_trained:       # petit corpus : entraîner sur tout
        all_v = np.vstack(buffer_vecs)
        index.train(all_v)
        flush(all_v, buffer_rows)
    con.commit()
    con.close()
    faiss.write_index(index, os.path.join(args.out, INDEX_FILE))
    t_index = time.time() - t0

    meta = {
        "embedder": spec, "dim": dim, "index_type": args.index, "n_docs": int(n_docs),
        "n_chunks": int(n_chunks), "chunk_size_words": size, "chunk_overlap_words": overlap,
        "sources_filter": args.sources or None, "spark_master": args.master,
        "partitions": args.partitions,
        "timings_s": {"ingestion": round(t_ingest, 2), "chunk_embed": round(t_embed, 2),
                      "indexing": round(t_index, 2), "total": round(time.time() - t_start, 2)},
        "embed_throughput_chunks_per_s": round(n_chunks / max(t_embed, 1e-9), 1),
    }
    with open(os.path.join(args.out, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nIndex prêt dans : {args.out}")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
_FTS_STOP = _STOP | set("what which who whom how did does do has have had can will would should about into than "
                        "then their there they them his her our your we you any all not but also each per".split())


def _fts_query(text: str) -> str:
    toks, seen = [], set()
    for t in re.findall(r"\w+", text.lower()):
        if len(t) > 2 and t not in _FTS_STOP and t not in seen:
            seen.add(t)
            toks.append(t)
    return " OR ".join(f'"{t}"' for t in toks[:40])


class Retriever:
    """Recherche par similarité cosinus (FAISS) + BM25 (SQLite FTS5) fusionnés par RRF ; texte lu dans SQLite.

    mode : "dense" (vecteurs seuls), "hybrid" (vecteurs + BM25), "auto" (hybride si l'index FTS existe,
           créé par la commande `ftsindex`).
    """

    def __init__(self, index_dir: str, embedder_spec: Optional[str] = None,
                 nprobe: int = 16, ef_search: int = 64, mode: Optional[str] = None):
        import faiss
        with open(os.path.join(index_dir, META_FILE), "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.index = faiss.read_index(os.path.join(index_dir, INDEX_FILE))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
        self.embedder = Embedder(embedder_spec or self.meta["embedder"])
        self.con = sqlite3.connect(os.path.join(index_dir, SQLITE_FILE), check_same_thread=False)
        self.has_fts = bool(self.con.execute(
            "SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone())
        mode = mode or os.environ.get("RAG_SEARCH_MODE", "auto")
        if mode == "hybrid" and not self.has_fts:
            raise RuntimeError("Mode hybride demandé mais index BM25 absent : lancer "
                               "`python spark_rag_pipeline.py ftsindex --index <dossier>`.")
        self.mode = "hybrid" if (mode == "hybrid" or (mode == "auto" and self.has_fts)) else "dense"

    def _fetch(self, ids: List[int]) -> Dict[int, tuple]:
        marks = ",".join("?" * len(ids))
        cur = self.con.execute(
            f"SELECT id, doc_id, section, chunk_id, n_chunks, source_type, text FROM chunks WHERE id IN ({marks})",
            ids)
        return {r[0]: r for r in cur.fetchall()}

    def _bm25(self, query: str, n: int) -> List[int]:
        q = _fts_query(query)
        if not q:
            return []
        try:
            cur = self.con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 3.0) LIMIT ?",
                (q, n))
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def search(self, queries: List[str], k: int = 5) -> List[List[dict]]:
        qv = self.embedder.encode_queries(queries)
        n_cand = max(k * 4, 50) if self.mode == "hybrid" else k
        scores, ids = self.index.search(qv, n_cand)
        results = []
        for qi in range(len(queries)):
            dense = [(int(i), float(sc)) for i, sc in zip(ids[qi], scores[qi]) if i >= 0]
            if self.mode == "hybrid":
                fused: Dict[int, float] = {}
                for rank, (i, _) in enumerate(dense, 1):                      # fusion RRF (constante 60)
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
                for rank, i in enumerate(self._bm25(queries[qi], n_cand), 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
                valid = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            else:
                valid = dense[:k]
            rows = self._fetch([i for i, _ in valid]) if valid else {}
            hits = []
            for rank, (i, sc) in enumerate(valid, 1):
                _, doc_id, section, chunk_id, n_chunks, src, text = rows[i]
                hits.append({"rank": rank, "score": sc, "doc_id": doc_id, "section": section,
                             "chunk_id": chunk_id, "n_chunks": n_chunks,
                             "metadata": {"source_type": src, "chunk_index": chunk_id}, "text": text})
            results.append(hits)
        return results

    @staticmethod
    def rank_documents(hits: List[dict]) -> List[str]:
        """Classement de documents (1re apparition d'un doc_id dans la liste de chunks)."""
        seen, ranked = set(), []
        for h in hits:
            if h["doc_id"] not in seen:
                seen.add(h["doc_id"])
                ranked.append(h["doc_id"])
        return ranked


def cmd_ftsindex(args):
    """Ajoute un index plein texte BM25 (SQLite FTS5) à un index existant : pas de recalcul d'embeddings."""
    t0 = time.time()
    con = sqlite3.connect(os.path.join(args.index, SQLITE_FILE))
    con.execute("DROP TABLE IF EXISTS chunks_fts")
    con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', "
                "content_rowid='id', tokenize='porter unicode61')")
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.commit()
    n = con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    con.close()
    print(f"Index BM25 créé : {n} chunks en {time.time() - t0:.0f}s. La recherche hybride est maintenant active.")


# --------------------------------------------------------------------------- #
# Génération (Ollama) et réponse avec citations
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are a technical documentation assistant for engineers.
Answer ONLY from the numbered context passages provided. Rules:
1. Reply in the same language as the question.
2. Give a structured answer as numbered steps (Step 1, Step 2, ...). Keep it concise and factual.
3. After every statement, cite its source passage(s) like [1] or [2][3]. Never invent a source.
4. If the passages do not contain the answer, reply with exactly: {NO_ANSWER}
5. If passages contradict each other, say so and cite both."""


class OllamaError(RuntimeError):
    pass


def ollama_chat(model: str, system: str, user: str, host: str = "http://localhost:11434",
                temperature: float = 0.0, num_ctx: int = 8192, timeout: int = 300) -> str:
    payload = {"model": model, "stream": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "options": {"temperature": temperature, "num_ctx": num_ctx}}
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, ConnectionError, OSError) as exc:
        raise OllamaError(f"Ollama indisponible ({exc}). Lancer 'ollama serve' et 'ollama pull {model}'.") from exc


def gemini_chat(model: str, system: str, user: str, temperature: float = 0.0, timeout: int = 0,
                retries: int = 3) -> str:
    """API Gemini (generateContent). Clé lue dans GEMINI_API_KEY ; délai via GEMINI_TIMEOUT (défaut 180 s)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise OllamaError("GEMINI_API_KEY non défini : le mettre dans le fichier .env")
    timeout = timeout or int(os.environ.get("GEMINI_TIMEOUT", "180"))
    base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
    payload = {"systemInstruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user}]}],
               "generationConfig": {"temperature": temperature}}
    req = urllib.request.Request(f"{base}/v1beta/models/{model}:generateContent",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    last = None
    for attempt in range(retries):
        retryable = True
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")[:200]
            last = f"HTTP {exc.code}: {body}"
            retryable = exc.code in (429, 500, 503)
        except (KeyError, IndexError, ValueError) as exc:     # réponse sans texte (ex. bloquée)
            last = f"réponse inattendue ({exc!r})"
            retryable = False
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = str(exc)                                    # délai dépassé, réseau : on réessaie
        if not retryable or attempt == retries - 1:
            break
        time.sleep(2 ** (attempt + 1))
    raise OllamaError(f"Gemini indisponible ({last}). Vérifier la clé, le nom du modèle et le quota.")


def llm_chat(provider: str, model: str, system: str, user: str, host: str = "http://localhost:11434") -> str:
    if provider == "gemini":
        return gemini_chat(model, system, user)
    return ollama_chat(model, system, user, host=host)


def build_prompt(question: str, hits: List[dict], max_chars: int = 1800) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        text = h["text"][:max_chars]
        blocks.append(f"[{n}] (source: {h['metadata']['source_type']} | title: {h['section']} | "
                      f"doc_id: {h['doc_id']})\n{text}")
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"


def answer_question(retriever: Retriever, question: str, k: int = 5, model: str = "llama3",
                    use_llm: bool = True, host: str = "http://localhost:11434",
                    provider: str = "ollama") -> dict:
    """Retrieval + génération. Retourne réponse, citations, latences, mode utilisé."""
    t0 = time.time()
    hits = retriever.search([question], k=k)[0]
    t_retr = time.time() - t0
    result = {"question": question, "hits": hits, "latency_retrieval_s": t_retr,
              "latency_generation_s": 0.0, "mode": "llm", "no_answer": False, "cited": []}
    if not hits:
        result.update(answer="Aucun passage trouvé.", no_answer=True, mode="none")
        return result
    if use_llm:
        try:
            t1 = time.time()
            answer = llm_chat(provider, model, SYSTEM_PROMPT, build_prompt(question, hits), host=host)
            result["latency_generation_s"] = time.time() - t1
            result["answer"] = answer
            result["no_answer"] = answer.strip().startswith(NO_ANSWER)
            result["cited"] = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)})
            return result
        except OllamaError as exc:
            result["llm_error"] = str(exc)
    # Repli extractif (LLM indisponible ou désactivé) : les passages sont la réponse
    steps = [f"Step {n}: {h['text'][:300].strip()}... [{n}]" for n, h in enumerate(hits[:3], 1)]
    result.update(mode="extractive", answer="\n".join(steps), cited=list(range(1, min(3, len(hits)) + 1)))
    return result


# --------------------------------------------------------------------------- #
# Commandes search / ask
# --------------------------------------------------------------------------- #
def cmd_search(args):
    retriever = Retriever(args.index, args.embedder)
    t0 = time.time()
    hits = retriever.search([args.query], k=args.k)[0]
    print(f"({(time.time() - t0) * 1000:.0f} ms)\n")
    for h in hits:
        print(f"#{h['rank']}  score={h['score']:.3f}  [{h['metadata']['source_type']}] "
              f"{h['section']}  (doc_id={h['doc_id']}, chunk {h['chunk_id']}/{h['n_chunks']})")
        print("    " + h["text"][:250].replace("\n", " ") + "...\n")


def cmd_ask(args):
    retriever = Retriever(args.index, args.embedder)
    res = answer_question(retriever, args.query, k=args.k, model=args.model, use_llm=not args.no_llm,
                          host=args.ollama_host, provider=args.provider)
    print(f"[mode={res['mode']}  retrieval={res['latency_retrieval_s'] * 1000:.0f} ms  "
          f"génération={res['latency_generation_s']:.1f} s]\n")
    print(res["answer"], "\n\nSources :")
    for n, h in enumerate(res["hits"], 1):
        flag = "*" if n in res["cited"] else " "
        print(f" {flag}[{n}] {h['metadata']['source_type']} | {h['section']} | doc_id={h['doc_id']} | "
              f"score={h['score']:.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Télécharger le dataset Hugging Face")
    d.add_argument("--repo", default=DEFAULT_REPO)
    d.add_argument("--out", default="data/enterprise_rag")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="Spark : chunking + embeddings, puis index FAISS")
    b.add_argument("--data", required=True, help="Dossier contenant les parquet du dataset")
    b.add_argument("--out", default="rag_index")
    b.add_argument("--master", default="local[*]", help="ex. local[4] ou spark://hote:7077")
    b.add_argument("--driver-memory", default="8g")
    b.add_argument("--partitions", type=int, default=16)
    b.add_argument("--limit-docs", type=int, default=0, help="0 = tout le corpus (sinon échantillon aléatoire)")
    b.add_argument("--sample-frac", type=float, default=0.0, help="fraction aléatoire du corpus, ex. 0.1")
    b.add_argument("--include-gold", default="", help="dossier des questions : ajoute leurs documents attendus "
                   "à l'échantillon (à déclarer dans le rapport : évaluation sur corpus réduit)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--reuse-embeddings", action="store_true",
                   help="sauter Spark et refaire seulement l'index à partir de <out>/_chunks_embeddings")
    b.add_argument("--prev-embed-seconds", type=float, default=0.0, help="(reprise) durée embeddings du run précédent")
    b.add_argument("--prev-ingest-seconds", type=float, default=0.0, help="(reprise) durée ingestion du run précédent")
    b.add_argument("--max-seq-length", type=int, default=256, help="tokens max par chunk (vitesse CPU)")
    b.add_argument("--torch-threads", type=int, default=0, help="threads PyTorch par worker (0 = défaut)")
    b.add_argument("--sources", default="", help="filtre source_type, ex. confluence,google_drive")
    b.add_argument("--chunk-size", type=int, default=150, help="mots par chunk (~200 tokens)")
    b.add_argument("--chunk-overlap", type=int, default=30)
    b.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    b.add_argument("--embed-batch", type=int, default=64)
    b.add_argument("--index", choices=["flat", "hnsw", "ivf", "ivfpq"], default="hnsw",
                   help="ivfpq = très économe en RAM (corpus complet)")
    b.add_argument("--train-size", type=int, default=100000, help="vecteurs pour entraîner IVF")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("ftsindex", help="Ajouter l'index BM25 (recherche hybride) à un index existant")
    f.add_argument("--index", default="rag_index")
    f.set_defaults(func=cmd_ftsindex)

    for name, fn in (("search", cmd_search), ("ask", cmd_ask)):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--index", default="rag_index")
        s.add_argument("--embedder", default=None, help="par défaut : celui de l'index")
        s.add_argument("-k", type=int, default=5)
        if name == "ask":
            s.add_argument("--provider", choices=["ollama", "gemini"], default=os.environ.get("RAG_PROVIDER", "ollama"))
            s.add_argument("--model", default=os.environ.get("RAG_LLM_MODEL", "llama3"), help="ex. llama3 (ollama) ou gemini-3.5-flash (gemini)")
            s.add_argument("--ollama-host", default="http://localhost:11434")
            s.add_argument("--no-llm", action="store_true", help="réponse extractive seulement")
        s.set_defaults(func=fn)

    args = p.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
