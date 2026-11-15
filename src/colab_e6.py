"""Бібліотека експерименту: корпус, представлення, BM25, злиття, метрики."""
import json, re, random
from functools import lru_cache
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42
TOK = re.compile(r"\w+(?:['’ʼ-]\w+)*", re.U)
SENT = re.compile(r'(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z"«(\d])')

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    ctxs = sorted({x["Context"] for x in d})
    cid = {c: i for i, c in enumerate(ctxs)}
    q = [x["Question"] for x in d]
    gold = np.array([cid[x["Context"]] for x in d])
    ans = [x["Answer"] for x in d]
    return ctxs, q, gold, ans

def split_by_context(ctxs, gold, dev_frac=0.3, seed=SEED):
    """Поділ запитань на dev/test за контекстами (без витоку між ними)."""
    rng = random.Random(seed)
    ids = list(range(len(ctxs))); rng.shuffle(ids)
    dev_ctx = set(ids[: int(dev_frac * len(ids))])
    dev = np.array([i for i, g in enumerate(gold) if g in dev_ctx])
    test = np.array([i for i, g in enumerate(gold) if g not in dev_ctx])
    return dev, test

def tok_surface(t): return [w.lower() for w in TOK.findall(t)]

_morph = None
@lru_cache(maxsize=None)
def lemma(w):
    global _morph
    if _morph is None:
        import pymorphy3; _morph = pymorphy3.MorphAnalyzer(lang="uk")
    return _morph.parse(w)[0].normal_form

def tok_lemma(t): return [lemma(w) for w in tok_surface(t)]
def tok_prefix(t, n=5): return [w[:n] for w in tok_surface(t)]
def tok_char(t, n=4):
    out = []
    for w in tok_surface(t):
        w = f"#{w}#"
        out += [w[i:i+n] for i in range(max(1, len(w)-n+1))]
    return out

VIEWS = {"surface": tok_surface, "lemma": tok_lemma, "prefix5": tok_prefix, "char4": tok_char}

class BM25:
    """Okapi BM25 (k1=1.2, b=0.75), IDF за Robertson–Sparck Jones зі згладжуванням (як у Lucene)."""
    def __init__(self, docs_tok, k1=1.2, b=0.75):
        self.cv = CountVectorizer(analyzer=lambda x: x)
        tf = self.cv.fit_transform(docs_tok).tocsr().astype(np.float64)
        N = tf.shape[0]
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
    def score(self, q_tok):
        Q = self.cv.transform(q_tok); Q.data[:] = 1.0  # бінарні терми запиту
        return (Q @ self.W.T).toarray()

def ranks_of_gold(S, gold):
    """1-based ранг золотого документа (лише строго кращі + 1; ties -> оптимістично не допускаємо)."""
    g = S[np.arange(len(gold)), gold]
    return (S > g[:, None]).sum(1) + 1 + 0 * g

def metrics(ranks):
    r = np.asarray(ranks)
    return {"R@1": float((r <= 1).mean()), "R@5": float((r <= 5).mean()),
            "R@10": float((r <= 10).mean()), "R@100": float((r <= 100).mean()),
            "MRR@10": float(np.where(r <= 10, 1 / r, 0).mean()),
            "nDCG@10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean())}

def mrr10_vec(ranks): r = np.asarray(ranks); return np.where(r <= 10, 1 / r, 0.0)

def minmax(S):
    lo = S.min(1, keepdims=True); hi = S.max(1, keepdims=True)
    return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)

def rank_matrix(S):
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order); rows = np.arange(S.shape[0])[:, None]
    R[rows, order] = np.arange(1, S.shape[1] + 1)[None, :]
    return R

def rrf(Ss, k=60):
    return sum(1.0 / (k + rank_matrix(S)) for S in Ss)

def cc(Ss, w):
    return sum(wi * minmax(S) for wi, S in zip(w, Ss))

def simplex_grid(m, step=0.1):
    n = int(round(1 / step)); out = []
    def rec(prefix, left, slots):
        if slots == 1: out.append(prefix + [left / n]); return
        for i in range(left + 1): rec(prefix + [i / n], left - i, slots - 1)
    rec([], n, m); return np.array(out)

def bootstrap_ci(x, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); x = np.asarray(x); n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired_bootstrap(a, b, B=5000, seed=SEED):
    """Різниця середніх (a-b), 95% ДІ та двосторонній p (частка бутстрепів зі знаком, протилежним спостережуваному)."""
    rng = np.random.default_rng(seed); d = np.asarray(a) - np.asarray(b); n = len(d)
    m = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(p)

# ---------------- Colab-частина: трансформерні ембединги + злиття ----------------
import os, sys, json, time, urllib.request, torch
if not os.path.exists("ua_squad_dataset.json"):
    urllib.request.urlretrieve("https://github.com/fido-ai/ua-datasets/releases/download/v0.0.1/ua_squad_dataset.json", "ua_squad_dataset.json")
ctxs, q, gold, ans = load("ua_squad_dataset.json")
dev, test = split_by_context(ctxs, gold)
assert len(ctxs) == 2620 and len(dev) == 4173 and len(test) == 9686, (len(ctxs), len(dev), len(test))
t0 = time.time(); S = {}
for n in ["surface", "lemma", "prefix5", "char4"]:
    f = VIEWS[n]; S[n] = BM25([f(c) for c in ctxs]).score([f(x) for x in q]).astype(np.float32)
print("lexical done", round(time.time() - t0), flush=True)
from sentence_transformers import SentenceTransformer
MODELS = {"e5-small": "intfloat/multilingual-e5-small", "e5-base": "intfloat/multilingual-e5-base", "bge-m3": "BAAI/bge-m3"}
out = {"env": {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
import sentence_transformers; out["env"]["sentence_transformers"] = sentence_transformers.__version__
out["lex_check"] = {n: metrics(ranks_of_gold(S[n][test], gold[test])) for n in S}
for tag, name in MODELS.items():
    t1 = time.time(); m = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu"); m.max_seq_length = 512
    e5 = "e5" in name
    D = m.encode([("passage: " if e5 else "") + c for c in ctxs], batch_size=32, normalize_embeddings=True)
    Q = m.encode([("query: " if e5 else "") + x for x in q], batch_size=128, normalize_embeddings=True)
    S[tag] = (Q @ D.T).astype(np.float32); del m; torch.cuda.empty_cache()
    print(tag, "encoded", round(time.time() - t1), flush=True)
def ev(Sm, idx): return ranks_of_gold(Sm[idx], gold[idx])
res = {"single": {}, "fusion": {}, "paired": {}}
R = {}
for n in S: R[n] = ev(S[n], test); res["single"][n] = metrics(R[n])
def fuse(names, step):
    Nd = [minmax(S[n][dev]) for n in names]; best, bw = -1, None
    for w in simplex_grid(len(names), step):
        mv = mrr10_vec(ranks_of_gold(sum(wi * x for wi, x in zip(w, Nd)), gold[dev])).mean()
        if mv > best: best, bw = mv, w
    r_cc = ranks_of_gold(cc([S[n][test] for n in names], bw), gold[test])
    r_rrf = ranks_of_gold(rrf([S[n][test] for n in names]), gold[test])
    r_eq = ranks_of_gold(cc([S[n][test] for n in names], np.ones(len(names)) / len(names)), gold[test])
    return {"names": names, "w": [float(x) for x in bw], "dev_mrr": float(best), "cc": metrics(r_cc), "rrf": metrics(r_rrf), "eq": metrics(r_eq)}, r_cc, r_rrf
f0, R["lex2:cc"], _ = fuse(["lemma", "char4"], 0.1); res["fusion"]["lex2"] = f0; print("lex2", f0["w"], flush=True)
f1, R["lex4:cc"], _ = fuse(["surface", "lemma", "prefix5", "char4"], 0.1); res["fusion"]["lex4"] = f1; print("lex4", f1["w"], flush=True)
for tag in MODELS:
    fa, R[f"lex2+{tag}:cc"], R[f"lex2+{tag}:rrf"] = fuse(["lemma", "char4", tag], 0.1); res["fusion"][f"lex2+{tag}"] = fa
    fb, R[f"lex4+{tag}:cc"], R[f"lex4+{tag}:rrf"] = fuse(["surface", "lemma", "prefix5", "char4", tag], 0.2); res["fusion"][f"lex4+{tag}"] = fb
    res["paired"][f"lex2+{tag}:cc vs lex2:cc"] = paired_bootstrap(mrr10_vec(R[f"lex2+{tag}:cc"]), mrr10_vec(R["lex2:cc"]))
    res["paired"][f"lex4+{tag}:cc vs lex4:cc"] = paired_bootstrap(mrr10_vec(R[f"lex4+{tag}:cc"]), mrr10_vec(R["lex4:cc"]))
    res["paired"][f"{tag} vs lemma"] = paired_bootstrap(mrr10_vec(R[tag]), mrr10_vec(R["lemma"]))
    print(tag, "fusion done", flush=True)
out["res"] = res; out["seconds"] = round(time.time() - t0)
s = json.dumps(out, ensure_ascii=False); open("e6_results.json", "w").write(s)
print("@@JSON_BEGIN@@"); print(s); print("@@JSON_END@@")
