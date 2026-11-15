"""Ембединги трансформерних моделей (запускати в Google Colab / Kaggle / локально).
Порядок документів і запитань — той самий, що в lib.load (унікальні контексти за алфавітом).
Використання:  pip install sentence-transformers ;  python embed_hf.py --data ua_squad_dataset.json --model intfloat/multilingual-e5-small
Результат: S_<модель>.npy — матриця косинусних оцінок [n_queries x n_docs], float32.
"""
import argparse, json, numpy as np, random, torch
from sentence_transformers import SentenceTransformer
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", default="intfloat/multilingual-e5-small")
ap.add_argument("--bs", type=int, default=64); a = ap.parse_args()
random.seed(42); np.random.seed(42); torch.manual_seed(42)
d = json.load(open(a.data, encoding="utf-8"))
ctxs = sorted({x["Context"] for x in d}); q = [x["Question"] for x in d]
e5 = "e5" in a.model.lower()
m = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu"); m.max_seq_length = 512
D = m.encode([("passage: " if e5 else "") + c for c in ctxs], batch_size=a.bs, normalize_embeddings=True, show_progress_bar=True)
Q = m.encode([("query: " if e5 else "") + x for x in q], batch_size=a.bs, normalize_embeddings=True, show_progress_bar=True)
name = a.model.split("/")[-1]
np.save(f"S_{name}.npy", (Q @ D.T).astype(np.float32)); print("saved", f"S_{name}.npy", Q.shape, D.shape)
import sentence_transformers; print("sentence-transformers", sentence_transformers.__version__, "torch", torch.__version__)
