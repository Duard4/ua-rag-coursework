"""Крок E4: відбір речень під бюджет токенів vs усічення топ-k документів."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
C = ROOT + "/cache"
ctxs, q, gold, ans = load(ROOT + "/data/ua_squad_dataset.json")
dev, test = split_by_context(ctxs, gold)
R = json.load(open("results/retrieval.json"))
names = R["fusion"]["all6"]["names"]; w = R["fusion"]["all6"]["cc_weights"]
S = [np.load(f"{C}/S_{n}.npy") for n in names]
ok = np.array([a.strip() != "" and a.lower() in ctxs[g].lower() for a, g in zip(ans, gold)])
dev_q = dev[ok[dev]]; test_q = test[ok[test]]
print("verbatim subset: dev", len(dev_q), "test", len(test_q), flush=True)
def fused(idx): return cc([s[idx] for s in S], w)
# речення
sents = [SENT.split(c) for c in ctxs]
flat = [(d, s) for d, ss in enumerate(sents) for s in ss]
sid = {}; 
for i, (d, s) in enumerate(flat): sid.setdefault(d, []).append(i)
bm = BM25([tok_lemma(s) for _, s in flat])
K = 5
def build(qi, top, docscore, B, lam, order, mode):
    """Повертає список (doc, sentence). Бюджет B — у словах."""
    if mode == "trunc":
        out, n = [], 0
        for d in top:
            for s in sents[d]:
                k = len(s.split())
                if n + k > B:
                    rem = B - n
                    if rem > 0: out.append((d, " ".join(s.split()[:rem])))
                    return out
                out.append((d, s)); n += k
        return out
    ids = [i for d in top for i in sid[d]]
    Qv = bm.cv.transform([tok_lemma(q[qi])]); Qv.data[:] = 1
    sc = np.asarray((Qv @ bm.W[ids].T).todense()).ravel()
    sc = sc / (sc.max() + 1e-9)
    prior = np.array([docscore[flat[i][0]] for i in ids])
    tot = sc + lam * prior
    out, n = [], 0
    for j in np.argsort(-tot, kind="stable"):
        s = flat[ids[j]][1]; k = len(s.split())
        if n + k <= B: out.append((flat[ids[j]][0], s, ids[j])); n += k
    if order == "orig": out.sort(key=lambda t: t[2])
    return [(d, s) for d, s, _ in out]
def run(idx, B, lam, order, mode, gold_only=False):
    F = fused(idx); cov = []; dis = []; hit = []
    for r, qi in enumerate(idx):
        if gold_only: top = [gold[qi]]; ds = {gold[qi]: 1.0}
        else:
            top = list(np.argsort(-F[r])[:K]); m = F[r][top]; ds = {d: (v - F[r].min()) / (F[r].max() - F[r].min() + 1e-9) for d, v in zip(top, m)}
        ctx = build(qi, top, ds, B, lam, order, mode)
        txt = " ".join(s for _, s in ctx).lower(); a = ans[qi].lower()
        cov.append(a in txt); nw = sum(len(s.split()) for _, s in ctx) or 1
        dis.append(sum(len(s.split()) for d, s in ctx if d != gold[qi]) / nw)
        hit.append(any(a in s.lower() for _, s in ctx))
    return np.array(cov, float), np.array(dis), np.array(hit, float)
out = {"K": K, "n_dev": int(len(dev_q)), "n_test": int(len(test_q)), "words_per_token_note": "бюджет у словах"}
# підбір lam на dev при B=80
best = None; out["lam_dev"] = {}
for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    c = run(dev_q, 80, lam, "best", "sent")[0].mean(); out["lam_dev"][lam] = float(c); print("dev lam", lam, round(c, 4), flush=True)
    if best is None or c > best[1]: best = (lam, c)
lam = best[0]; out["lam"] = lam
out["test"] = {}; per = {}
for B in [40, 80, 120, 200, 400]:
    row = {}
    for name, kw in {"trunc_topk": dict(mode="trunc", lam=0, order="orig"),
                     "sent_best_first": dict(mode="sent", lam=lam, order="best"),
                     "sent_orig_order": dict(mode="sent", lam=lam, order="orig"),
                     "sent_no_prior": dict(mode="sent", lam=0.0, order="best"),
                     "oracle_gold_doc_trunc": dict(mode="trunc", lam=0, order="orig", gold_only=True)}.items():
        c, dsh, h = run(test_q, B, **kw); per[(B, name)] = c
        row[name] = {"coverage": bootstrap_ci(c), "distractor_share": float(dsh.mean()), "sentence_hit": float(h.mean())}
        print(B, name, [round(x, 4) for x in row[name]["coverage"]], round(row[name]["distractor_share"], 3), flush=True)
    out["test"][B] = row
out["paired"] = {f"B={B}": paired_bootstrap(per[(B, "sent_best_first")], per[(B, "trunc_topk")]) for B in [40, 80, 120, 200, 400]}
out["paired_order"] = {f"B={B}": paired_bootstrap(per[(B, "sent_best_first")], per[(B, "sent_orig_order")]) for B in [40, 80, 120, 200, 400]}
json.dump(out, open("results/context.json", "w"), ensure_ascii=False, indent=1)
