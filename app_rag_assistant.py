#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_rag_assistant.py - Interface terrain de l'assistant RAG (Gradio + FastAPI)

Fonctionnalités
---------------
* Réponse structurée étape par étape, avec citations [n] générées par le LLM (LLaMA 3 / Ollama)
* Tableau des sources : type de source, titre/section, doc_id, chunk, score, extrait
* Affichage du passage complet de chaque source pour validation par l'ingénieur
* Mode extractif automatique si Ollama est indisponible (le système reste utilisable)
* API REST : POST /api/ask  {"question": "...", "k": 5}

Lancement
---------
    python app_rag_assistant.py --index rag_index --model llama3 --port 7860
    -> interface : http://localhost:7860      API : http://localhost:7860/api/ask
"""
import argparse
import os
import re
from typing import List

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from spark_rag_pipeline import NO_ANSWER, Retriever, answer_question

STATE = {"retriever": None, "model": os.environ.get("RAG_LLM_MODEL", "llama3"),
         "host": "http://localhost:11434", "provider": os.environ.get("RAG_PROVIDER", "ollama")}


def get_retriever() -> Retriever:
    if STATE["retriever"] is None:
        STATE["retriever"] = Retriever(os.environ.get("RAG_INDEX", "rag_index"),
                                       os.environ.get("RAG_EMBEDDER") or None)
    return STATE["retriever"]


def format_answer(res: dict) -> str:
    """Met en forme la réponse du LLM : étapes numérotées + bandeau de mode/latence."""
    banner = (f"*Mode : **{res['mode']}** · retrieval {res['latency_retrieval_s'] * 1000:.0f} ms · "
              f"génération {res['latency_generation_s']:.1f} s*")
    if res["mode"] == "extractive":
        banner += ("\n\n> ⚠️ Le LLM est indisponible : voici directement les passages les plus pertinents "
                   f"(`{res.get('llm_error', 'LLM désactivé')}`).")
    if res.get("no_answer"):
        body = ("**Information introuvable** dans les documents indexés. "
                "Reformulez la question ou vérifiez la source directement.")
    else:
        body = re.sub(r"(?m)^\s*(Step|Étape)\s+(\d+)\s*[:.\-]?", r"**\1 \2 :**", res["answer"])
    return f"{body}\n\n{banner}"


def sources_table(res: dict) -> List[list]:
    rows = []
    for n, h in enumerate(res["hits"], 1):
        rows.append([("✔ " if n in res["cited"] else "") + f"[{n}]", h["metadata"]["source_type"], h["section"],
                     h["doc_id"], f"{h['chunk_id'] + 1}/{h['n_chunks']}", round(h["score"], 3),
                     h["text"][:220].replace("\n", " ") + "…"])
    return rows


def sources_details(res: dict) -> str:
    parts = []
    for n, h in enumerate(res["hits"], 1):
        parts.append(f"### [{n}] {h['section']}\n`{h['metadata']['source_type']}` · doc_id `{h['doc_id']}` · "
                     f"chunk {h['chunk_id'] + 1}/{h['n_chunks']} · score {h['score']:.3f}\n\n> "
                     + h["text"].replace("\n", "\n> "))
    return "\n\n".join(parts)


def ask(question: str, k: int, use_llm: bool, model: str):
    if not question or not question.strip():
        return "Saisissez une question.", [], ""
    res = answer_question(get_retriever(), question.strip(), k=int(k), model=model or STATE["model"],
                          use_llm=use_llm, host=STATE["host"], provider=STATE["provider"])
    return format_answer(res), sources_table(res), sources_details(res)


# --------------------------------------------------------------------------- #
# Interface Gradio
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Assistant IA - Ingénierie documentaire") as demo:
        gr.Markdown("# 🔧 Assistant IA & RAG distribué\n"
                    "Posez une question technique : la réponse est construite **uniquement** à partir des "
                    "documents indexés, avec citation des sources à valider.")
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question de l'ingénieur", lines=3,
                                      placeholder="Ex. What is the p99 latency target for service svc-abcde?")
                with gr.Row():
                    k = gr.Slider(1, 10, value=5, step=1, label="Passages récupérés (top-k)")
                    model = gr.Textbox(value=STATE["model"], label="Modèle LLM")
                    use_llm = gr.Checkbox(value=True, label="Génération LLM")
                btn = gr.Button("Répondre", variant="primary")
            with gr.Column(scale=4):
                answer = gr.Markdown(label="Réponse")
        gr.Markdown("### Sources citées (✔ = citée dans la réponse)")
        table = gr.Dataframe(headers=["#", "Source", "Titre / section", "doc_id", "Chunk", "Score", "Extrait"],
                             wrap=True, interactive=False)
        with gr.Accordion("Passages complets (validation terrain)", open=False):
            details = gr.Markdown()
        inputs, outputs = [question, k, use_llm, model], [answer, table, details]
        btn.click(ask, inputs, outputs)
        question.submit(ask, inputs, outputs)
    return demo


# --------------------------------------------------------------------------- #
# API REST (FastAPI) + UI montée sur /
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str
    k: int = 5
    use_llm: bool = True


api = FastAPI(title="RAG Assistant API")


@api.get("/api/health")
def health():
    meta = get_retriever().meta
    return {"status": "ok", "n_chunks": meta["n_chunks"], "embedder": meta["embedder"]}


@api.post("/api/ask")
def api_ask(req: AskRequest):
    res = answer_question(get_retriever(), req.question, k=req.k, model=STATE["model"],
                          use_llm=req.use_llm, host=STATE["host"], provider=STATE["provider"])
    return {"answer": res["answer"], "mode": res["mode"], "no_answer": res["no_answer"],
            "latency_retrieval_s": res["latency_retrieval_s"], "latency_generation_s": res["latency_generation_s"],
            "sources": [{"n": n, "cited": n in res["cited"], "doc_id": h["doc_id"], "title": h["section"],
                         "source_type": h["metadata"]["source_type"], "chunk": h["chunk_id"],
                         "score": h["score"], "text": h["text"]} for n, h in enumerate(res["hits"], 1)]}


app = gr.mount_gradio_app(api, build_ui(), path="/")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="rag_index")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--provider", choices=["ollama", "gemini"], default=os.environ.get("RAG_PROVIDER", "ollama"))
    ap.add_argument("--model", default=os.environ.get("RAG_LLM_MODEL", "llama3"), help="ex. llama3.2:3b (ollama) ou gemini-3.5-flash (gemini)")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    os.environ["RAG_INDEX"] = a.index
    if a.embedder:
        os.environ["RAG_EMBEDDER"] = a.embedder
    STATE.update(model=a.model, host=a.ollama_host, provider=a.provider)
    uvicorn.run(app, host=a.host, port=a.port)
