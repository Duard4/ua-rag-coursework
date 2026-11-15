"""Кроки E1–E3: порівняння представлень, злиття RRF vs CC, ефективність за обсягом dev."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
C = ROOT + "/cache"
ctxs, q, gold, ans = load(ROOT + "/data/ua_squad_dataset.json")
dev, test = split_by_context(ctxs, gold)
S = {n: np.load(f"{C}/S_{n}.npy") for n in ["surface", "lemma", "prefix5", "char4", "lsa", "dense"]}
res = {"n_docs": len(ctxs), "n_queries": len(q), "n_dev": len(dev), "n_test": len(test)}
R = {}
def ev(Sm, idx): return ranks_of_gold(Sm[idx], gold[idx])
# E1/E2: окремі системи на test
res["single"] = {}
for n, Sm in S.items():
    r = ev(Sm, test); R[n] = r; res["single"][n] = metrics(r)
    print(n, {k: round(v, 4) for k, v in res["single"][n].items()})
# Злиття: набір систем
sets = {"lex3": ["surface", "lemma", "char4"], "lex+dense": ["lemma", "char4", "dense"],
        "all6": ["surface", "lemma", "prefix5", "char4", "lsa", "dense"], "lemma+dense": ["lemma", "dense"]}
res["fusion"] = {}
for sn, names in sets.items():
    Ss = [S[n] for n in names]
    r_rrf = ev_r = ranks_of_gold(rrf([s[test] for s in Ss]), gold[test])
    # ваги CC підбираються на dev (крок 0.1 симплекса) за MRR@10
    Sd = [s[dev] for s in Ss]; Nd = [minmax(s) for s in Sd]
    best, bw = -1, None
    for w in simplex_grid(len(Ss), 0.1 if len(Ss) <= 4 else 0.2):
        m = mrr10_vec(ranks_of_gold(sum(wi * x for wi, x in zip(w, Nd)), gold[dev])).mean()
        if m > best: best, bw = m, w
    r_cc = ranks_of_gold(cc([s[test] for s in Ss], bw), gold[test])
    r_eq = ranks_of_gold(cc([s[test] for s in Ss], np.ones(len(Ss)) / len(Ss)), gold[test])
    R[f"{sn}:rrf"], R[f"{sn}:cc"], R[f"{sn}:eq"] = r_rrf, r_cc, r_eq
    res["fusion"][sn] = {"names": names, "cc_weights": [float(x) for x in bw], "dev_mrr": float(best),
        "rrf": metrics(r_rrf), "cc_tuned": metrics(r_cc), "cc_equal": metrics(r_eq)}
    print(sn, "w=", bw, {k: round(res["fusion"][sn][k]["MRR@10"], 4) for k in ["rrf", "cc_tuned", "cc_equal"]})
# Чутливість RRF до k (перевірка коректності)
res["rrf_k"] = {}
for k in [1, 5, 10, 30, 60]:
    r = ranks_of_gold(rrf([S[n][test] for n in sets["lex3"]], k=k), gold[test]); res["rrf_k"][k] = metrics(r)
    print("rrf k=", k, round(res["rrf_k"][k]["MRR@10"], 4), flush=True)
# Значущість (парний бутстреп, MRR@10 на test)
sig = {}
pairs = [("lemma", "surface"), ("char4", "lemma"), ("dense", "lemma"),
         ("all6:cc", "all6:rrf"), ("all6:cc", "lemma"), ("all6:rrf", "lemma"),
         ("lex+dense:cc", "lex+dense:rrf"), ("lex+dense:cc", "lemma"), ("lemma+dense:cc", "lemma")]
for a, b in pairs:
    sig[f"{a} vs {b}"] = paired_bootstrap(mrr10_vec(R[a]), mrr10_vec(R[b]))
    print(a, "vs", b, [round(x, 4) for x in sig[f"{a} vs {b}"]])
res["paired_bootstrap_mrr10"] = sig
res["ci_mrr10"] = {k: bootstrap_ci(mrr10_vec(v)) for k, v in R.items() if ":" not in k or k.endswith(":cc") or k.endswith(":rrf")}
# Ефективність за обсягом dev (all6): 30 повторів
rng = np.random.default_rng(SEED); names = sets["all6"]; Ss = [S[n] for n in names]
Nall = [minmax(s) for s in Ss]  # мін-макс по рядках, тож можна індексувати
grid = simplex_grid(len(Ss), 0.2)  # крок 0.2 для швидкості
# попередньо: для кожної точки сітки MRR на кожному dev-запиті (вектор) -> швидкий підбір за підвибіркою
mrr_dev = np.stack([mrr10_vec(ranks_of_gold(sum(wi * x[dev] for wi, x in zip(w, Nall)), gold[dev])) for w in grid])
mrr_test = np.stack([mrr10_vec(ranks_of_gold(sum(wi * x[test] for wi, x in zip(w, Nall)), gold[test])) for w in grid])
rrf_test = float(mrr10_vec(R["all6:rrf"]).mean()); eq_test = float(mrr10_vec(R["all6:eq"]).mean())
eff = {}
for n in [10, 25, 50, 100, 300, 1000, len(dev)]:
    vals = []
    for _ in range(30):
        sub = rng.choice(len(dev), n, replace=False)
        vals.append(float(mrr_test[mrr_dev[:, sub].mean(1).argmax()].mean()))
    eff[n] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals))}
    print("dev n=", n, eff[n])
res["sample_efficiency"] = {"grid_step": 0.2, "repeats": 30, "rrf_test": rrf_test, "equal_test": eq_test,
    "oracle_grid_test": float(mrr_test.mean(1).max()), "by_n": eff}
# Лише частка запитань, де злиття покращило / погіршило порівняно з лемами
res["all6_cc_vs_lemma_wins"] = {"better": int((R["all6:cc"] < R["lemma"]).sum()), "worse": int((R["all6:cc"] > R["lemma"]).sum())}
json.dump(res, open("results/retrieval.json", "w"), ensure_ascii=False, indent=1)
np.save(ROOT + "/cache/dev_idx.npy", dev); np.save(ROOT + "/cache/test_idx.npy", test)
