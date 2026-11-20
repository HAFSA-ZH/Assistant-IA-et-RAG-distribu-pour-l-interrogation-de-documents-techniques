#!/usr/bin/env python3
"""Génère un mini-dataset SYNTHÉTIQUE au même schéma qu'EnterpriseRAG-Bench
(documents + questions) pour tester le code sans télécharger 1,4 Go.

    python make_sample_data.py --out data/sample --docs 600 --questions 60

Ce n'est PAS le vrai benchmark : les métriques obtenues dessus ne sont qu'un test
de bon fonctionnement du pipeline.
"""
import argparse
import os
import random

import pandas as pd

FILLER = ("The team reviewed the quarterly roadmap and aligned on priorities. Several action items were "
          "recorded and assigned during the weekly sync. Customers reported minor issues that were triaged "
          "by support. The platform continues to scale across regions while keeping costs under control. "
          "Documentation was updated to reflect the latest process changes and onboarding guidelines. ").split()


def filler(rng, n):
    return " ".join(rng.choice(FILLER) for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    sources = ["confluence", "slack", "gmail", "jira", "google_drive", "github"]
    people = ["Amina Berrada", "Lucas Martin", "Priya Nair", "Omar Haddad", "Elena Rossi", "Chen Wei"]
    docs, facts = [], []
    for i in range(args.docs):
        src = sources[i % len(sources)]
        code = "svc-" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        owner, slo, region = rng.choice(people), rng.choice([120, 180, 240, 300, 450]), \
            rng.choice(["eu-west", "us-east", "ap-south"])
        fact = (f"The p99 latency target for service {code} is {slo} ms and it is owned by {owner}. "
                f"It is deployed in region {region}.")
        pos = rng.randint(20, 160)
        content = filler(rng, pos) + " " + fact + " " + filler(rng, rng.randint(80, 400))
        doc_id = f"doc_{i:05d}"
        docs.append({"doc_id": doc_id, "source_type": src, "title": f"{code} runbook ({src})", "content": content})
        facts.append((doc_id, src, code, owner, slo, region))

    qs = []
    for q in range(args.questions):
        doc_id, src, code, owner, slo, region = rng.choice(facts)
        if q % 10 == 9:   # question sans réponse dans le corpus
            qs.append({"question_id": f"q{q:03d}", "question_type": "information_not_found",
                       "source_types": [], "question": f"What is the p99 latency target of service svc-zzzzq{q}?",
                       "expected_doc_ids": [], "gold_answer": "Not available in the corpus.",
                       "answer_facts": []})
        elif q % 2 == 0:
            qs.append({"question_id": f"q{q:03d}", "question_type": "simple", "source_types": [src],
                       "question": f"What is the p99 latency target for service {code}?",
                       "expected_doc_ids": [doc_id], "gold_answer": f"{slo} ms",
                       "answer_facts": [f"{slo} ms"]})
        else:
            qs.append({"question_id": f"q{q:03d}", "question_type": "semantic", "source_types": [src],
                       "question": f"Who owns {code} and in which region is it deployed?",
                       "expected_doc_ids": [doc_id], "gold_answer": f"{owner}, region {region}",
                       "answer_facts": [owner, region]})
    pd.DataFrame(docs).to_parquet(os.path.join(args.out, "documents.parquet"), index=False)
    pd.DataFrame(qs).to_parquet(os.path.join(args.out, "questions.parquet"), index=False)
    print(f"{len(docs)} documents et {len(qs)} questions écrits dans {args.out}")


if __name__ == "__main__":
    main()
