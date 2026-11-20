# Mini-Projet 05 - Assistant IA & RAG distribué (EnterpriseRAG-Bench)

Spark (chunking + embeddings distribués) -> FAISS + SQLite -> LLaMA 3 (Ollama) -> interface Gradio/FastAPI.

## Fichiers
| Fichier | Rôle |
|---|---|
| `spark_rag_pipeline.py` | Téléchargement, ingestion Spark, chunking, embeddings, index FAISS, retrieval, génération |
| `app_rag_assistant.py` | Interface Gradio + API FastAPI (`/api/ask`), citations des sources |
| `evaluate_rag.py` | Recall@k, MRR, nDCG, latence, métriques de génération |
| `make_report.py` | Produit `rag_evaluation_report.pdf` à partir de `results.json` |
| `make_sample_data.py` | Mini-dataset synthétique pour tester sans télécharger 1,4 Go |
| `rag_evaluation_report.pdf` | Rapport **de validation** (données synthétiques) - à régénérer avec le vrai dataset |

## Prérequis
Python 3.10+, **Java 17+** (Spark), `pip install -r requirements.txt`.
LLM : installer [Ollama](https://ollama.com), puis `ollama pull llama3`.

## Exécution sur le vrai dataset
```bash
python spark_rag_pipeline.py download --out data/enterprise_rag
# essai sur un sous-ensemble (recommandé d'abord)
python spark_rag_pipeline.py build --data data/enterprise_rag --out rag_index --limit-docs 20000
# corpus complet
python spark_rag_pipeline.py build --data data/enterprise_rag --out rag_index --partitions 64 --driver-memory 16g --index hnsw
python evaluate_rag.py --index rag_index --data data/enterprise_rag --generate --n-gen 50 --out results.json
python make_report.py results.json --out rag_evaluation_report.pdf
python app_rag_assistant.py --index rag_index --model llama3     # http://localhost:7860
```

## Test rapide sans téléchargement ni modèle
```bash
python make_sample_data.py --out data/sample
python spark_rag_pipeline.py build --data data/sample --out rag_index_test --embedder hash --partitions 4
python evaluate_rag.py --index rag_index_test --data data/sample --smoke-test
python app_rag_assistant.py --index rag_index_test --embedder hash
```
L'encodeur `hash` n'est qu'un outil de test ; utiliser `st:BAAI/bge-small-en-v1.5` (défaut) pour les vrais résultats.

## Points d'attention
* Les noms de fichiers/colonnes du dataset sont **auto-détectés** (documents : `doc_id`+`content` ;
  questions : `question`+`expected_doc_ids`). Si le schéma réel diffère, adapter `find_document_files` / `load_questions`.
* Mémoire : le corpus complet donne plusieurs millions de chunks. Utiliser `--index ivf` ou un GPU pour les embeddings,
  et un cluster (`--master spark://...`) pour accélérer.
* Questions en français : utiliser `--embedder st:intfloat/multilingual-e5-small` (reconstruire l'index).
* Ne pas utiliser le corpus du benchmark pour entraîner un modèle.
