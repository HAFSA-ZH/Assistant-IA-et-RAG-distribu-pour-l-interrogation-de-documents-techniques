#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_rag.py - Évaluation du RAG sur les questions d'EnterpriseRAG-Bench

Mesures
-------
Retrieval (niveau document, via expected_doc_ids) :
    Recall@k, HitRate@k, MRR, nDCG@10, par type de question
Latence :
    retrieval (requête unique) : moyenne, p50, p95
Génération (optionnelle, --generate, nécessite Ollama) :
    couverture des faits (answer_facts), F1 de tokens vs gold_answer,
    précision/rappel des citations, taux de refus correct / faux refus,
    latence de génération.
    NB : ce sont des PROXYS automatiques. Le score officiel du benchmark
    utilise un LLM-juge (correction x complétude).

Usage
-----
    python evaluate_rag.py --index rag_index --data data/enterprise_rag --out results.json
    python evaluate_rag.py --index rag_index --data data/enterprise_rag --generate --model llama3 --n-gen 50
    python make_report.py results.json          # -> rag_evaluation_report.pdf
"""
import argparse
import glob
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from spark_rag_pipeline import NO_ANSWER, Retriever, answer_question

KS = (1, 3, 5, 10, 20)
STOP = set("a an the and or of to in on for with by at from as is are was were be it its this that and".split())
NOT_FOUND_HINTS = ("not_found", "not found", "notfound", "introuvable", "unanswerable", "missing", "no_answer")


# --------------------------------------------------------------------------- #
# Chargement des questions (schéma auto-détecté)
# --------------------------------------------------------------------------- #
def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(v) for v in json.loads(s)]
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in re.split(r"[;,|]", s) if v.strip()]


def load_questions(data_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)):
        try:
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq
                if not {"question", "expected_doc_ids"} <= set(pq.ParquetFile(path).schema_arrow.names):
                    continue                      # ex. fichier de documents de 1,4 Go : on ne le charge pas
                df = pd.read_parquet(path)
            elif path.endswith(".jsonl"):
                df = pd.read_json(path, lines=True)
            elif path.endswith(".json"):
                df = pd.read_json(path)
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                continue
        except Exception:
            continue
        if {"question", "expected_doc_ids"} <= set(df.columns):
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Aucun fichier de questions (colonnes question/expected_doc_ids) sous {data_dir}")
    q = pd.concat(frames, ignore_index=True)
    q["expected"] = q["expected_doc_ids"].map(_as_list)
    q["facts"] = q["answer_facts"].map(_as_list) if "answer_facts" in q else [[] for _ in range(len(q))]
    if "question_type" not in q:
        q["question_type"] = "unknown"
    if "gold_answer" not in q:
        q["gold_answer"] = ""
    # "Sans réponse" = défini par le TYPE de question (ex. info_not_found).
    # Une question répondable peut ne pas avoir de documents attendus (ex. high_level,
    # réponse diffuse dans le corpus) : elle reste répondable mais sans vérité terrain de retrieval.
    has_type = "question_type" in q and q["question_type"].astype(str).str.lower().ne("unknown").any()
    if has_type:
        q["not_found"] = [any(h in str(t).lower() for h in NOT_FOUND_HINTS) for t in q["question_type"]]
    else:
        q["not_found"] = [len(e) == 0 for e in q["expected"]]
    q["has_gt"] = [len(e) > 0 for e in q["expected"]]
    return q


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def retrieval_scores(ranked_docs: List[str], expected: List[str]) -> Dict[str, float]:
    exp = set(expected)
    out = {}
    for k in KS:
        found = len(exp & set(ranked_docs[:k]))
        out[f"recall@{k}"] = found / len(exp)
        out[f"hit@{k}"] = float(found > 0)
    rr = next((1.0 / r for r, d in enumerate(ranked_docs, 1) if d in exp), 0.0)
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked_docs[:10], 1) if d in exp)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(exp), 10) + 1))
    out["mrr"] = rr
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", (text or "").lower()) if t not in STOP]


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    common = defaultdict(int)
    for t in g:
        common[t] += 1
    overlap = 0
    for t in p:
        if common[t] > 0:
            common[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def fact_coverage(answer: str, facts: List[str], thr: float = 0.7) -> float:
    if not facts:
        return float("nan")
    ans = set(_tokens(answer))
    covered = 0
    for f in facts:
        ft = _tokens(f)
        if ft and sum(t in ans for t in ft) / len(ft) >= thr:
            covered += 1
    return covered / len(facts)


def mean(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def evaluate(args) -> dict:
    questions = load_questions(args.data)
    if args.max_questions:
        questions = questions.sample(n=min(args.max_questions, len(questions)), random_state=0) \
                             .reset_index(drop=True)
    retriever = Retriever(args.index, args.embedder, nprobe=args.nprobe, ef_search=args.ef_search,
                          mode=args.search_mode)
    print(f"{len(questions)} questions ; index = {retriever.meta['n_chunks']} chunks")

    # -- Retrieval : qualité (requêtes en lot) -------------------------------
    k_chunks = max(args.chunks_for_docs, max(KS))
    all_hits = []
    for i in range(0, len(questions), 64):
        all_hits.extend(retriever.search(list(questions["question"][i:i + 64]), k=k_chunks))

    per_q, by_type = [], defaultdict(list)
    for (_, row), hits in zip(questions.iterrows(), all_hits):
        if row["not_found"] or not row["has_gt"]:
            continue          # pas de vérité terrain de retrieval pour cette question
        ranked = retriever.rank_documents(hits)
        sc = retrieval_scores(ranked, row["expected"])
        per_q.append({"question_id": row.get("question_id"), "type": row["question_type"],
                      "question": row["question"], "expected": row["expected"],
                      "top_docs": ranked[:5], **sc})
        by_type[row["question_type"]].append(sc)
    metric_names = [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + ["mrr", "ndcg@10"]
    overall = {m: mean([q[m] for q in per_q]) for m in metric_names}
    types = {t: {"n": len(v), **{m: mean([x[m] for x in v]) for m in metric_names}} for t, v in by_type.items()}

    # -- Retrieval : latence (requête unique, après chauffe) ----------------
    sample_q = list(questions["question"])[:args.latency_queries]
    for q in sample_q[:3]:
        retriever.search([q], k=args.k)
    lat = []
    for q in sample_q:
        t0 = time.perf_counter()
        retriever.search([q], k=args.k)
        lat.append((time.perf_counter() - t0) * 1000)
    latency = {"n_queries": len(lat), "mean_ms": float(np.mean(lat)), "p50_ms": pct(lat, 50),
               "p95_ms": pct(lat, 95), "max_ms": float(np.max(lat)), "k": args.k}

    results = {"config": {"index": args.index, "embedder": retriever.embedder.spec, "k": args.k,
                          "n_questions_total": int(len(questions)),
                          "n_questions_answerable": len(per_q),
                          "n_questions_not_found": int(questions["not_found"].sum()),
                          "n_questions_no_ground_truth": int((~questions["has_gt"] & ~questions["not_found"]).sum()),
                          "nprobe": args.nprobe, "ef_search": args.ef_search, "search_mode": retriever.mode},
               "index_meta": retriever.meta, "retrieval": {"overall": overall, "by_type": types},
               "latency_retrieval": latency,
               "worst_retrieval": sorted(per_q, key=lambda x: (x["recall@10"], x["mrr"]))[:15],
               "smoke_test": bool(args.smoke_test)}

    # -- Génération (optionnelle) ---------------------------------------------
    if args.generate:
        subset = questions.sample(n=min(args.n_gen, len(questions)), random_state=1)
        rows, failed = [], 0
        for _, row in subset.iterrows():
            if args.gen_sleep:
                time.sleep(args.gen_sleep)
            res = answer_question(retriever, row["question"], k=args.k, model=args.model,
                                  use_llm=True, host=args.ollama_host, provider=args.provider)
            if res["mode"] != "llm":
                failed += 1                       # on continue : une erreur ne doit pas tout arrêter
                print(f"LLM indisponible ({failed} échec(s)) :", res.get("llm_error", "?")[:160])
                continue
            cited_docs = {res["hits"][n - 1]["doc_id"] for n in res["cited"]}
            exp = set(row["expected"])
            rows.append({
                "question_id": row.get("question_id"), "type": row["question_type"],
                "not_found": bool(row["not_found"]), "no_answer": res["no_answer"],
                "fact_coverage": fact_coverage(res["answer"], row["facts"]),
                "f1": token_f1(res["answer"], row["gold_answer"]),
                "cite_precision": (len(cited_docs & exp) / len(cited_docs)) if cited_docs and exp else float("nan"),
                "cite_recall": (len(cited_docs & exp) / len(exp)) if exp else float("nan"),
                "has_citation": bool(res["cited"]),
                "lat_retrieval_s": res["latency_retrieval_s"], "lat_generation_s": res["latency_generation_s"]})
        if rows:
            ans = [r for r in rows if not r["not_found"]]
            nf = [r for r in rows if r["not_found"]]
            tot = [r["lat_retrieval_s"] + r["lat_generation_s"] for r in rows]
            results["generation"] = {
                "model": f"{args.provider}:{args.model}", "n_failed": failed, "n": len(rows), "n_answerable": len(ans), "n_not_found": len(nf),
                "fact_coverage": mean([r["fact_coverage"] for r in ans]),
                "token_f1": mean([r["f1"] for r in ans]),
                "citation_rate": mean([float(r["has_citation"]) for r in ans]),
                "citation_precision": mean([r["cite_precision"] for r in ans]),
                "citation_recall": mean([r["cite_recall"] for r in ans]),
                "false_refusal_rate": mean([float(r["no_answer"]) for r in ans]),
                "correct_refusal_rate": mean([float(r["no_answer"]) for r in nf]) if nf else float("nan"),
                "latency_generation_mean_s": mean([r["lat_generation_s"] for r in rows]),
                "latency_total_p50_s": pct(tot, 50), "latency_total_p95_s": pct(tot, 95)}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="rag_index")
    ap.add_argument("--data", required=True, help="dossier contenant les questions du dataset")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("-k", type=int, default=5, help="passages donnés au LLM")
    ap.add_argument("--chunks-for-docs", type=int, default=50, help="chunks récupérés pour classer les documents")
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--latency-queries", type=int, default=100)
    ap.add_argument("--search-mode", choices=["auto", "dense", "hybrid"], default="auto",
                    help="dense = vecteurs seuls ; hybrid = vecteurs + BM25 (nécessite `ftsindex`)")
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--generate", action="store_true", help="évaluer aussi la génération (Ollama)")
    ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--gen-sleep", type=float, default=0.0, help="pause (s) entre appels LLM (quota)")
    ap.add_argument("--provider", choices=["ollama", "gemini"], default=os.environ.get("RAG_PROVIDER", "ollama"))
    ap.add_argument("--model", default=os.environ.get("RAG_LLM_MODEL", "llama3"))
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--smoke-test", action="store_true", help="marque le rapport comme test sur données synthétiques")
    args = ap.parse_args()

    results = evaluate(args)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    o = results["retrieval"]["overall"]
    print(json.dumps({k: round(v, 3) for k, v in o.items()}, indent=2))
    print("latence retrieval :", {k: round(v, 1) for k, v in results["latency_retrieval"].items()})
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
