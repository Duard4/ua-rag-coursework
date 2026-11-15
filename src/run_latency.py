"""Час відповіді на один запит (CPU, 1 потік Python): пошук за 4 лексичними представленнями + злиття + відбір речень."""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
ctxs, q, gold, ans = load(ROOT + "/data/ua_squad_dataset.json")
dev, test = split_by_context(ctxs, gold)
R = json.load(open("results/retrieval.json")); w = R["fusion"]["lex3"]["cc_weights"]
t0 = time.time(); idx = {n: BM25([f(c) for c in ctxs]) for n, f in VIEWS.items() if n in ("surface", "lemma", "char4")}; index_s = time.time() - t0
sents = [SENT.split(c) for c in ctxs]; flat = [(d, s) for d, ss in enumerate(sents) for s in ss]
sid = {}
for i, (d, s) in enumerate(flat): sid.setdefault(d, []).append(i)
t0 = time.time(); bms = BM25([tok_lemma(s) for _, s in flat]); sent_index_s = time.time() - t0
rng = np.random.default_rng(SEED); sample = rng.choice(test, 500, replace=False)
lat = {"retrieve": [], "select": []}
for qi in sample:
    t = time.perf_counter()
    S = [idx[n].score([VIEWS[n](q[qi])])[0:1] for n in ("surface", "lemma", "char4")]
    F = cc(S, w)[0]; top = np.argsort(-F)[:5]
    lat["retrieve"].append(time.perf_counter() - t)
    t = time.perf_counter()
    ids = [i for d in top for i in sid[d]]
    Qv = bms.cv.transform([tok_lemma(q[qi])]); Qv.data[:] = 1
    sc = np.asarray((Qv @ bms.W[ids].T).todense()).ravel(); _ = np.argsort(-sc)
    lat["select"].append(time.perf_counter() - t)
out = {"index_build_s_3views": index_s, "sentence_index_s": sent_index_s, "n_sentences": len(flat), "n_queries_timed": 500,
       **{f"{k}_ms_mean": float(np.mean(v) * 1000) for k, v in lat.items()}, **{f"{k}_ms_p95": float(np.percentile(v, 95) * 1000) for k, v in lat.items()}}
json.dump(out, open("results/latency.json", "w"), indent=1); print(out)
