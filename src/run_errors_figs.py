"""E5: аналіз помилок і побудова рисунків."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ROOT + "/cache"
ctxs, q, gold, ans = load(ROOT + "/data/ua_squad_dataset.json")
dev, test = split_by_context(ctxs, gold)
R = json.load(open("results/retrieval.json")); X = json.load(open("results/context.json"))
names = R["fusion"]["all6"]["names"]; w = R["fusion"]["all6"]["cc_weights"]
Sm = cc([np.load(f"{C}/S_{n}.npy")[test] for n in names], w)
r = ranks_of_gold(Sm, gold[test])
fail = r > 10; err = {"n_test": int(len(test)), "fail_R10": int(fail.sum()), "fail_share": float(fail.mean())}
qi_all = test
ov = np.array([len(set(tok_lemma(q[i])) & set(tok_lemma(ctxs[gold[i]]))) for i in qi_all])
qlen = np.array([len(tok_surface(q[i])) for i in qi_all])
err["zero_lemma_overlap_share_among_fail"] = float((ov[fail] == 0).mean())
err["zero_lemma_overlap_share_among_ok"] = float((ov[~fail] == 0).mean())
err["median_qlen_fail"] = float(np.median(qlen[fail])); err["median_qlen_ok"] = float(np.median(qlen[~fail]))
err["median_overlap_fail"] = float(np.median(ov[fail])); err["median_overlap_ok"] = float(np.median(ov[~fail]))
# кореляція «контекстів на документ» і провалу: скільки запитань мають документ із >8 запитань
cnt = np.bincount(gold); err["fail_share_docs_ge8_q"] = float(fail[cnt[gold[test]] >= 8].mean()); err["fail_share_docs_lt8_q"] = float(fail[cnt[gold[test]] < 8].mean())
answerable = np.array([a.strip() != "" for a in ans])[test]
err["n_test_answerable"] = int(answerable.sum()); err["n_empty_answer_total"] = int(sum(a.strip() == "" for a in ans))
err["mrr10_answerable_only"] = float(mrr10_vec(r[answerable]).mean()); err["mrr10_unanswerable_only"] = float(mrr10_vec(r[~answerable]).mean())
rng = np.random.default_rng(SEED); ex = rng.choice(np.where(fail & answerable)[0], 4, replace=False)
err["examples"] = [{"q": q[test[i]], "gold_rank": int(r[i]), "overlap": int(ov[i]), "answer": ans[test[i]]} for i in ex]
json.dump(err, open("results/errors.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(err, ensure_ascii=False, indent=1))
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
# Рис. 1
lab = {"surface": "поверхневі\nформи", "lemma": "леми", "prefix5": "префікс-5", "char4": "симв. 4-грами", "lsa": "LSA-300", "dense": "floret-\nвектори"}
v = [R["single"][k]["MRR@10"] for k in lab]
fig, ax = plt.subplots(figsize=(6.5, 3.2)); b = ax.bar(list(lab.values()), v, color="#4c72b0")
for x, y in zip(b, v): ax.text(x.get_x() + x.get_width() / 2, y + .01, f"{y:.3f}", ha="center", fontsize=8)
ax.set_ylabel("MRR@10 (test)"); ax.set_ylim(0, .9); plt.tight_layout(); plt.savefig("figures/fig1_views.png"); plt.close()
# Рис. 2: RRF k vs CC
ks = [1, 5, 10, 30, 60]; fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(ks, [R["rrf_k"][str(k)]["MRR@10"] for k in ks], "o-", label="RRF (лем+симв.+пов.)")
ax.axhline(R["fusion"]["lex3"]["cc_tuned"]["MRR@10"], color="g", ls="--", label="CC, ваги з dev")
ax.axhline(R["single"]["char4"]["MRR@10"], color="gray", ls=":", label="найкраща одиночна (4-грами)")
ax.set_xlabel("параметр k у RRF"); ax.set_ylabel("MRR@10 (test)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/fig2_rrf_k.png"); plt.close()
# Рис. 3: ефективність за розміром dev
se = R["sample_efficiency"]["by_n"]; ns = sorted(map(int, se)); m = np.array([se[str(n)]["mean"] for n in ns]); s = np.array([se[str(n)]["std"] for n in ns])
fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.errorbar(ns, m, yerr=s, fmt="o-", capsize=3, label="CC: ваги з n запитань dev")
ax.axhline(R["sample_efficiency"]["rrf_test"], color="r", ls="--", label="RRF, k=60"); ax.axhline(R["sample_efficiency"]["equal_test"], color="orange", ls=":", label="рівні ваги")
ax.set_xscale("log"); ax.set_xlabel("кількість розмічених запитань n"); ax.set_ylabel("MRR@10 (test)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/fig3_sample_eff.png"); plt.close()
# Рис. 4: покриття vs бюджет
Bs = [40, 80, 120, 200, 400]; fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, l, st in [("trunc_topk", "усічення топ-5 документів", "s-"), ("sent_best_first", "відбір речень (запропонований)", "o-"), ("sent_no_prior", "відбір речень без пріоритету документа", "^--")]:
    ax.plot(Bs, [X["test"][str(B)][k]["coverage"][0] for B in Bs], st, label=l)
ax.set_xlabel("бюджет контексту, слів"); ax.set_ylabel("покриття відповіді"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/fig4_budget.png"); plt.close()
