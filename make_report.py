#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py - Génère rag_evaluation_report.pdf à partir de results.json

    python make_report.py results.json --out rag_evaluation_report.pdf

Le rapport est produit à partir des mesures réelles de evaluate_rag.py :
aucun chiffre n'est écrit à la main.
"""
import argparse
import datetime
import json
import math
import os
import tempfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                Preformatted)

KS = (1, 3, 5, 10, 20)
NAVY = colors.HexColor("#1F3A5F")


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def table(data, col_widths=None, header=True, font=8.5):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")])]
    t.setStyle(TableStyle(style))
    return t


def build(results: dict, out_path: str, baseline: dict = None, note: str = "", src_name: str = "results.json"):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13.5, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, textColor=NAVY, spaceBefore=10, spaceAfter=6, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=4, keepWithNext=1)
    title = ParagraphStyle("title", parent=ss["Title"], fontSize=21, textColor=NAVY)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    warn = ParagraphStyle("warn", parent=body, backColor=colors.HexColor("#FFF3CD"), borderColor=colors.HexColor("#E0B100"),
                          borderWidth=0.8, borderPadding=6, textColor=colors.HexColor("#5C4400"))

    cfg, meta = results["config"], results["index_meta"]
    ov, bt = results["retrieval"]["overall"], results["retrieval"]["by_type"]
    lat, gen = results["latency_retrieval"], results.get("generation")
    tmp = tempfile.mkdtemp()
    story = []

    # ---- Titre -----------------------------------------------------------
    story += [Paragraph("Rapport d'évaluation du système RAG distribué", title),
              Paragraph("Mini-Projet 05 - Assistant IA &amp; RAG distribué (Spark, FAISS, LLaMA 3 / Ollama)",
                        ParagraphStyle("sub", parent=body, alignment=1, fontSize=11)),
              Paragraph(f"Généré le {datetime.date.today():%d/%m/%Y} à partir de {escape(src_name)}", 
                        ParagraphStyle("d", parent=small, alignment=1)), Spacer(1, 10)]
    if results.get("smoke_test"):
        story += [Paragraph("<b>RAPPORT DE VALIDATION - DONNÉES SYNTHÉTIQUES.</b> Les chiffres ci-dessous proviennent du "
                            "jeu synthétique <i>make_sample_data.py</i> avec un encodeur par hachage ; ils prouvent que "
                            "la chaîne fonctionne de bout en bout mais <b>ne reflètent pas</b> les performances sur "
                            "EnterpriseRAG-Bench. Pour le vrai rapport : lancer <i>build</i> puis <i>evaluate_rag.py</i> "
                            "sur le dataset réel, puis relancer <i>make_report.py</i>.", warn), Spacer(1, 6)]

    # ---- 1. Contexte -------------------------------------------------------
    story += [Paragraph("1. Objectif et périmètre", h1),
              Paragraph("Le projet consiste à concevoir un système de génération augmentée par récupération (RAG) "
                        "distribué capable d'indexer un grand volume de documentation et de répondre aux requêtes "
                        "d'ingénieurs avec citation des sources. Ce rapport évalue la qualité de la récupération, la "
                        "pertinence des réponses et la latence.", body),
              Paragraph("<b>Corpus visé :</b> EnterpriseRAG-Bench (~512 000 documents d'une entreprise fictive : Slack, "
                        "Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira, Confluence ; 500 questions "
                        "annotées). Le cas d'usage est élargi de la documentation technique de terrain à la "
                        "documentation d'entreprise ; les champs <i>page</i> et <i>appareil</i> du format de chunk "
                        "initial n'existent pas dans ce corpus et sont remplacés par <i>source_type</i> et <i>titre</i>.",
                        body)]

    # ---- 2. Architecture -----------------------------------------------
    story += [Paragraph("2. Architecture du pipeline", h1)]
    arch = ("Parquet (documents)\n"
            "   |  Spark : lecture, filtrage, repartition\n"
            "   v\n"
            "Chunking distribué (mapInPandas)  -> fenêtres de mots avec recouvrement\n"
            "   v\n"
            "Embeddings distribués (un modèle chargé par partition) -> vecteurs L2-normalisés\n"
            "   v\n"
            "FAISS (similarité cosinus = produit scalaire)  +  SQLite (texte et métadonnées des chunks)\n"
            "   v\n"
            "Question -> embedding -> top-k chunks -> prompt augmenté -> LLM (Ollama) -> réponse + citations [n]")
    if cfg.get("search_mode") == "hybrid":
        arch = arch.replace("FAISS (similarité cosinus = produit scalaire)  +  SQLite (texte et métadonnées des chunks)",
                            "FAISS (cosinus) + SQLite FTS5 (BM25) -> fusion RRF  ;  SQLite = texte + métadonnées")
    story += [Preformatted(arch, ParagraphStyle("pre", fontName="Courier", fontSize=8, leading=10.5,
                                                backColor=colors.HexColor("#F2F5F9"), borderPadding=6)),
              Spacer(1, 6),
              Paragraph("Format de chunk : <font face='Courier'>{doc_id, section(=titre), chunk_id, "
                        "metadata:{source_type, chunk_index}, text}</font>. La traçabilité s'appuie sur "
                        "source_type + titre + doc_id + numéro de chunk, affichés dans l'interface avec "
                        "l'extrait complet pour validation par l'ingénieur.", body)]

    # ---- 3. Configuration -------------------------------------------------
    t = meta["timings_s"]
    story += [Paragraph("3. Configuration expérimentale et coût d'indexation", h1),
              table([["Paramètre", "Valeur"],
                     ["Encodeur d'embeddings", escape(str(cfg["embedder"]))],
                     ["Dimension des vecteurs", str(meta["dim"])],
                     ["Type d'index FAISS", str(meta["index_type"])],
                     ["Mode de recherche", {"dense": "vectorielle seule", "hybrid": "hybride (vecteurs + BM25, fusion RRF)"}.get(cfg.get("search_mode", "dense"), "vectorielle seule")],
                     ["Documents indexés", f"{meta['n_docs']:,}"],
                     ["Chunks indexés", f"{meta['n_chunks']:,}"],
                     ["Taille / recouvrement des chunks", f"{meta['chunk_size_words']} / {meta['chunk_overlap_words']} mots"],
                     ["Spark (master, partitions)", f"{meta['spark_master']}, {meta['partitions']}"],
                     ["Filtre de sources", str(meta.get('sources_filter') or "aucun")],
                     ["Questions : retrieval évalué (avec documents attendus)", f"{cfg['n_questions_answerable']}"],
                     ["Questions sans réponse dans le corpus", f"{cfg['n_questions_not_found']}"],
                     ["Questions répondables sans document attendu", f"{cfg.get('n_questions_no_ground_truth', 0)}"]],
                    [7 * cm, 9.5 * cm]),
              Spacer(1, 6)]
    reduced = meta["n_docs"] < 400000 and not results.get("smoke_test")
    if reduced or note:
        txt = note or ""
        if reduced:
            txt = (f"<b>Corpus réduit.</b> L'index contient {meta['n_docs']:,} documents sur ~511 962 : les scores ne sont pas "
                   "directement comparables à une évaluation sur le corpus complet. " + txt)
        story += [Paragraph(txt, warn), Spacer(1, 6)]
    story += [table([["Étape", "Durée (s)"], ["Ingestion Spark", f"{t['ingestion']}"],
                     ["Chunking + embeddings (Spark)", f"{t['chunk_embed']}"], ["Indexation FAISS + SQLite", f"{t['indexing']}"],
                     ["Total (somme des étapes)", f"{round(t['ingestion'] + t['chunk_embed'] + t['indexing'], 1)}"],
                     ["Débit d'embedding (chunks/s)", f"{meta['embed_throughput_chunks_per_s']}"]], [7 * cm, 9.5 * cm])]

    # ---- 4. Retrieval -------------------------------------------------------
    story += [Paragraph("4. Qualité de la récupération", h1),
              Paragraph("Les métriques sont calculées au niveau document : les chunks récupérés sont dédupliqués par "
                        "doc_id (rang = première apparition) puis comparés à <i>expected_doc_ids</i>. "
                        "Recall@k = part des documents attendus présents dans les k premiers documents ; "
                        "HitRate@k = au moins un document attendu ; MRR = inverse du rang du premier document correct ; "
                        "nDCG@10 avec pertinence binaire. Les questions sans réponse dans le corpus sont exclues de ce "
                        "calcul et traitées dans la section génération.", body)]
    rows = [["Métrique"] + [f"@{k}" for k in KS],
            ["Recall"] + [f3(ov[f"recall@{k}"]) for k in KS],
            ["HitRate"] + [f3(ov[f"hit@{k}"]) for k in KS]]
    story += [table(rows, [4 * cm] + [2.4 * cm] * len(KS)), Spacer(1, 4),
              Paragraph(f"<b>MRR = {f3(ov['mrr'])}</b> · <b>nDCG@10 = {f3(ov['ndcg@10'])}</b>", body)]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(KS, [ov[f"recall@{k}"] for k in KS], marker="o", label="Recall@k", color="#1F3A5F")
    ax.plot(KS, [ov[f"hit@{k}"] for k in KS], marker="s", linestyle="--", label="HitRate@k", color="#D9822B")
    ax.set_xlabel("k (documents)"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    ax.set_xticks(KS)
    fig.tight_layout(); p1 = os.path.join(tmp, "recall.png"); fig.savefig(p1, dpi=170); plt.close(fig)
    story += [Image(p1, width=12.5 * cm, height=5.8 * cm)]

    if bt:
        story += [Paragraph("Résultats par type de question", h2)]
        rows = [["Type", "n", "Recall@5", "Recall@10", "MRR", "nDCG@10"]]
        for name, v in sorted(bt.items(), key=lambda kv: -kv[1]["n"]):
            rows.append([escape(str(name)), str(v["n"]), f3(v["recall@5"]), f3(v["recall@10"]), f3(v["mrr"]), f3(v["ndcg@10"])])
        story += [table(rows, [5.5 * cm, 1.5 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])]

    # ---- Comparaison avant/après (recherche vectorielle seule vs hybride) ----
    if baseline:
        bo, bl = baseline["retrieval"]["overall"], baseline["latency_retrieval"]
        bmode = {"dense": "Vectorielle seule", "hybrid": "Hybride"}
        n_mode = bmode.get(cfg.get("search_mode", "dense"), "Nouveau")
        b_mode = bmode.get(baseline["config"].get("search_mode", "dense"), "Référence")
        story += [Paragraph("Comparaison avec la recherche de référence", h2),
                  Paragraph(f"Référence : <b>{b_mode}</b> ; version évaluée : <b>{n_mode}</b>.", body)]
        rows = [["Métrique", b_mode, n_mode, "Écart"]]
        for key, label in (("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("recall@10", "Recall@10"),
                           ("hit@10", "HitRate@10"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10")):
            rows.append([label, f3(bo[key]), f3(ov[key]), f"{ov[key] - bo[key]:+.3f}"])
        rows.append(["Latence moyenne (ms)", f"{bl['mean_ms']:.0f}", f"{lat['mean_ms']:.0f}",
                     f"{lat['mean_ms'] - bl['mean_ms']:+.0f}"])
        story += [table(rows, [5 * cm, 3.6 * cm, 3.6 * cm, 3 * cm])]
        types = [t for t in bt if t in baseline["retrieval"]["by_type"]]
        if types:
            fig, ax = plt.subplots(figsize=(6.2, 2.8))
            xs = range(len(types)); w = 0.38
            ax.bar([x - w / 2 for x in xs], [baseline["retrieval"]["by_type"][t]["recall@10"] for t in types],
                   w, label=b_mode, color="#9AA5B8")
            ax.bar([x + w / 2 for x in xs], [bt[t]["recall@10"] for t in types], w, label=n_mode, color="#1F3A5F")
            ax.set_xticks(list(xs)); ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=6)
            ax.set_ylabel("Recall@10"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(); p2 = os.path.join(tmp, "cmp.png"); fig.savefig(p2, dpi=170); plt.close(fig)
            story += [Spacer(1, 4), Image(p2, width=13.5 * cm, height=6.1 * cm)]

    # ---- 5. Latence ---------------------------------------------------------
    story += [Paragraph("5. Latence", h1),
              Paragraph(f"Mesurée sur {lat['n_queries']} requêtes isolées (après 3 requêtes de chauffe), top-{lat['k']}, "
                        "incluant l'embedding de la requête, la recherche FAISS et la lecture SQLite.", body),
              table([["Moyenne", "p50", "p95", "Max"],
                     [f"{lat['mean_ms']:.1f} ms", f"{lat['p50_ms']:.1f} ms", f"{lat['p95_ms']:.1f} ms", f"{lat['max_ms']:.1f} ms"]],
                    [4 * cm] * 4)]
    if gen:
        story += [Spacer(1, 4), Paragraph(f"Génération ({escape(gen['model'])}) : latence moyenne "
                                           f"{gen['latency_generation_mean_s']:.2f} s ; latence totale p50 "
                                           f"{gen['latency_total_p50_s']:.2f} s, p95 {gen['latency_total_p95_s']:.2f} s.", body)]

    # ---- 6. Génération --------------------------------------------------------
    story += [Paragraph("6. Qualité des réponses générées", h1)]
    if gen:
        story += [Paragraph(f"Échantillon de {gen['n']} questions ({gen['n_answerable']} avec réponse, {gen['n_not_found']} "
                            "sans réponse). Mesures automatiques approchées (proxys) ; le score officiel du benchmark repose "
                            "sur un LLM-juge.", body),
                  table([["Indicateur", "Valeur"],
                         ["Appels LLM échoués (exclus des moyennes)", str(gen.get("n_failed", 0))],
                         ["Couverture des faits attendus (answer_facts)", f3(gen["fact_coverage"])],
                         ["F1 de tokens vs réponse de référence", f3(gen["token_f1"])],
                         ["Part de réponses avec citation", f3(gen["citation_rate"])],
                         ["Précision des citations (docs cités corrects)", f3(gen["citation_precision"])],
                         ["Rappel des citations", f3(gen["citation_recall"])],
                         [f"Faux refus (questions avec réponse, n={gen['n_answerable']})", f3(gen["false_refusal_rate"])],
                         [f"Refus correct (questions sans réponse, n={gen['n_not_found']})", f3(gen["correct_refusal_rate"])]], [10 * cm, 4 * cm])]
    else:
        story += [Paragraph("Non évaluée dans cette exécution (option <i>--generate</i> non activée ou Ollama "
                            "indisponible). Lancer : <font face='Courier'>python evaluate_rag.py --generate --model llama3 "
                            "--n-gen 50</font>.", body)]

    # ---- 7. Analyse d'erreurs ---------------------------------------------------
    worst = [w for w in results.get("worst_retrieval", []) if w["recall@10"] < 1][:8]
    story += [Paragraph("7. Analyse d'erreurs (retrieval)", h1)]
    if worst:
        rows = [["Type", "Question", "Recall@10", "MRR"]]
        for w in worst:
            q = w["question"] if len(w["question"]) < 90 else w["question"][:87] + "..."
            rows.append([escape(str(w["type"])), Paragraph(escape(q), ParagraphStyle("c", fontSize=7.5, leading=9)),
                         f3(w["recall@10"]), f3(w["mrr"])])
        story += [table(rows, [3 * cm, 9.5 * cm, 2 * cm, 2 * cm], font=8)]
    else:
        story += [Paragraph("Aucune question en échec sur l'échantillon évalué.", body)]

    # ---- 8. Limites -----------------------------------------------------------------
    story += [Paragraph("8. Limites et pistes d'amélioration", h1),
              Paragraph(("- La recherche hybride (BM25 + vecteurs) est déjà active ; les pistes restantes sont un reranker "
                         "(cross-encoder) et un encodeur d'embeddings plus grand (ex. bge-base)."
                         if cfg.get("search_mode") == "hybrid" else
                         "- Récupération purement dense : une recherche hybride (BM25 + vecteurs) et un reranker "
                         "(cross-encoder) amélioreraient les questions à contraintes et celles portant sur des identifiants "
                         "précis.") +
                        "<br/>- Chunking à taille fixe en mots : un découpage sensible à la structure (messages Slack, "
                        "fils d'e-mails, sections Confluence) préserverait mieux le contexte.<br/>- Questions "
                        "multi-documents (exhaustivité, vue d'ensemble) : augmenter k, reformuler la requête ou "
                        "ajouter une étape d'agrégation.<br/>- Les corpus et questions sont en anglais : pour des "
                        "questions en français, utiliser un encodeur multilingue (ex. multilingual-e5).<br/>- Les métriques "
                        "de génération sont des approximations lexicales calculées sur un petit échantillon ; un LLM-juge et "
                        "plus de questions donneraient un score comparable au classement officiel.<br/>- Le corpus du "
                        "benchmark ne doit pas servir à entraîner un modèle.", body),
              Paragraph("9. Reproduction", h1),
              Preformatted("python spark_rag_pipeline.py download --out data/enterprise_rag\n"
                           "python spark_rag_pipeline.py build --data data/enterprise_rag --out rag_index_50k \\\n"
                           "    --limit-docs 50000 --include-gold data/enterprise_rag\n"
                           "python spark_rag_pipeline.py ftsindex --index rag_index_50k\n"
                           "python evaluate_rag.py --index rag_index_50k --data data/enterprise_rag \\\n"
                           "    --search-mode hybrid --generate --out results_final.json\n"
                           "python make_report.py results_final.json --baseline results.json \\\n"
                           "    --out rag_evaluation_report.pdf\n"
                           "python app_rag_assistant.py --index rag_index_50k --provider ollama --model llama3.2:3b",
                           ParagraphStyle("pre2", fontName="Courier", fontSize=8, leading=10.5,
                                          backColor=colors.HexColor("#F2F5F9"), borderPadding=6)),
              Spacer(1, 8),
              Paragraph("Référence du jeu de données : EnterpriseRAG-Bench (Hugging Face, licence MIT ; arXiv 2605.05253).",
                        small)]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, "Assistant IA & Engineering RAG - Spark & LLM")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); canvas.restoreState()

    SimpleDocTemplate(out_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=2 * cm, title="Rapport d'évaluation RAG").build(story, onFirstPage=footer,
                                                                                    onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results.json")
    ap.add_argument("--out", default="rag_evaluation_report.pdf")
    ap.add_argument("--note", default="", help="texte libre ajouté sous la configuration (ex. description du corpus réduit)")
    ap.add_argument("--baseline", default=None, help="results.json de référence (ex. recherche vectorielle seule)")
    a = ap.parse_args()
    with open(a.results, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    base = None
    if a.baseline:
        with open(a.baseline, "r", encoding="utf-8") as fh:
            base = json.load(fh)
    build(res, a.out, base, a.note, os.path.basename(a.results))
    print("Rapport écrit :", a.out)


if __name__ == "__main__":
    main()
