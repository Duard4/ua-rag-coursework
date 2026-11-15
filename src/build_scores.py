"""Крок 1: матриці оцінок «запит × документ» для всіх представлень (кеш у ../cache)."""
import sys, time, os, numpy as np, json
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
C = ROOT + "/cache"; os.makedirs(C, exist_ok=True)
ctxs, q, gold, ans = load(ROOT + "/data/ua_squad_dataset.json")
t0 = time.time(); log = {}
for name, f in VIEWS.items():
    d_t = [f(c) for c in ctxs]; q_t = [f(x) for x in q]
    S = BM25(d_t).score(q_t).astype(np.float32)
    np.save(f"{C}/S_{name}.npy", S); log[name] = time.time() - t0; print(name, log[name], flush=True)
# LSA над TF-IDF лем
d_t = [tok_lemma(c) for c in ctxs]; q_t = [tok_lemma(x) for x in q]
tv = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True); X = tv.fit_transform(d_t)
svd = TruncatedSVD(300, random_state=SEED); D = svd.fit_transform(X); Qv = svd.transform(tv.transform(q_t))
nz = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
np.save(f"{C}/S_lsa.npy", (nz(Qv) @ nz(D).T).astype(np.float32)); print("lsa", time.time() - t0, flush=True)
# Щільні вектори: floret spaCy, усереднення з вагами IDF
import spacy
nlp = spacy.load("uk_core_news_lg", disable=["tagger", "morphologizer", "parser", "attribute_ruler", "lemmatizer", "ner"]) if True else None
idf = dict(zip(tv.get_feature_names_out() if hasattr(tv, "get_feature_names_out") else [], []))
bm = BM25([tok_surface(c) for c in ctxs]); vocab = bm.cv.vocabulary_; IDF = bm.idf
def embed(texts):
    out = np.zeros((len(texts), nlp.vocab.vectors_length), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        ws, vs = [], []
        for t in doc:
            if t.is_punct or t.is_space: continue
            w = IDF[vocab[t.lower_]] if t.lower_ in vocab else IDF.max()
            ws.append(w); vs.append(t.vector)
        if ws: out[i] = (np.array(ws)[:, None] * np.array(vs)).sum(0) / sum(ws)
    return out
Dv = embed(ctxs); Qd = embed(q)
np.save(f"{C}/S_dense.npy", (nz(Qd) @ nz(Dv).T).astype(np.float32)); print("dense", time.time() - t0, flush=True)
print("vectors_length", nlp.vocab.vectors_length, "n_vectors", nlp.vocab.vectors.shape)
