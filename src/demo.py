"""Інтерактивна демонстрація курсової роботи (CPU, ~2 хв).

Показує два модулі методу на реальних запитаннях UA-SQuAD:
  M1 — гібридний пошук: BM25 за поверхневими формами, лемами й символьними 4-грамами,
       нормалізовані оцінки об'єднуються опуклою комбінацією (CC) з вагами з results/retrieval.json;
  M2 — формування контексту під бюджет слів: відбір речень (λ = 4) проти простого усічення топ-5 документів.
Генерація відповіді мовною моделлю в роботі не розглядається, тому для кожного контексту
перевіряється, чи містить він еталонну відповідь (покриття відповіді).

Запуск:   python src/demo.py            (потрібні data/ua_squad_dataset.json і results/retrieval.json)
Команди:  Enter — приклад, де усічення втрачає відповідь, а відбір речень її зберігає
          будь-який текст — власне запитання (корпус — статті Вікіпедії зі SQuAD: буддизм,
          Шварценеггер, Канье Вест, Плімут тощо; про інші теми абзаців немає)
          b 80 — змінити бюджет контексту (слів);   q — вийти
"""
import json, os, sys, time, textwrap
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (ROOT, load, split_by_context, VIEWS, BM25, minmax, SENT, tok_lemma)

K, LAM = 5, 4.0


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------- підготовка ----------
t0 = time.time()
data = os.path.join(ROOT, "data", "ua_squad_dataset.json")
if not os.path.exists(data):
    sys.exit(f"Немає {data}. Завантажте набір: див. README (ua_squad_dataset.json).")
ctxs, q, gold, ans = load(data)
dev, test = split_by_context(ctxs, gold)
try:
    F = json.load(open(os.path.join(ROOT, "results", "retrieval.json")))["fusion"]["lex3"]
    NAMES, WEIGHTS = F["names"], F["cc_weights"]
except Exception:
    NAMES, WEIGHTS = ["surface", "lemma", "char4"], [0.0, 0.4, 0.6]
print("Будуємо індекси BM25 за представленнями:", ", ".join(n for n, w in zip(NAMES, WEIGHTS) if w > 0), flush=True)
INDEX = {n: BM25([VIEWS[n](c) for c in ctxs]) for n, w in zip(NAMES, WEIGHTS) if w > 0}
sents = [SENT.split(c) for c in ctxs]
flat = [(d, s) for d, ss in enumerate(sents) for s in ss]
sid = {}
for i, (d, s) in enumerate(flat):
    sid.setdefault(d, []).append(i)
bm_s = BM25([tok_lemma(s) for _, s in flat])
ok = np.array([a.strip() != "" and a.lower() in ctxs[g].lower() for a, g in zip(ans, gold)])
test_ans = test[ok[test]]            # тестові запитання з відповіддю, що дослівно є в абзаці
print(f"Готово за {time.time() - t0:.0f} с: {len(ctxs)} абзаців, {len(flat)} речень, "
      f"ваги CC {dict(zip(NAMES, WEIGHTS))}", flush=True)


# ---------- M1: пошук ----------
def retrieve(text):
    fused = sum(w * minmax(INDEX[n].score([VIEWS[n](text)])) for n, w in zip(NAMES, WEIGHTS) if w > 0)[0]
    top = list(np.argsort(-fused)[:K])
    lo, hi = fused.min(), fused.max()
    prior = {d: (fused[d] - lo) / (hi - lo + 1e-9) for d in top}
    return top, fused, prior


# ---------- M2: контекст ----------
def truncate(top, B):
    out, n = [], 0
    for d in top:
        for s in sents[d]:
            k = len(s.split())
            if n + k > B:
                if B - n > 0:
                    out.append((d, " ".join(s.split()[:B - n])))
                return out
            out.append((d, s)); n += k
    return out


def select(text, top, prior, B):
    ids = [i for d in top for i in sid[d]]
    Qv = bm_s.cv.transform([tok_lemma(text)]); Qv.data[:] = 1
    sc = np.asarray((Qv @ bm_s.W[ids].T).todense()).ravel(); sc = sc / (sc.max() + 1e-9)
    tot = sc + LAM * np.array([prior[flat[i][0]] for i in ids])
    out, n = [], 0
    for j in np.argsort(-tot, kind="stable"):
        s = flat[ids[j]][1]; k = len(s.split())
        if n + k <= B:
            out.append((flat[ids[j]][0], s)); n += k
    return out


def show_ctx(title, ctx, answer):
    txt = " ".join(s for _, s in ctx)
    words = len(txt.split())
    mark = ""
    if answer is not None:
        mark = "  ->  ВІДПОВІДЬ Є ✓" if answer.lower() in txt.lower() else "  ->  відповіді немає ✗"
    print(f"\n{title} ({words} слів){mark}")
    for d, s in ctx:
        print(textwrap.fill(f"[док. {d}] {s}", 100, initial_indent="  ", subsequent_indent="      "))


def run(text, B, answer=None, gold_doc=None):
    t = time.time()
    top, fused, prior = retrieve(text)
    ms = (time.time() - t) * 1000
    banner(f"ЗАПИТАННЯ: {text}")
    if answer is not None:
        print(f"Еталонна відповідь: «{answer}»")
    print(f"\nM1. Топ-{K} абзаців після злиття ({ms:.0f} мс):")
    for r, d in enumerate(top, 1):
        tag = "  <- золотий абзац" if gold_doc is not None and d == gold_doc else ""
        print(f"  {r}. док. {d}  оцінка {fused[d]:.3f}{tag}  | {textwrap.shorten(ctxs[d], 70)}")
    if gold_doc is not None and gold_doc not in top:
        print("  (золотого абзацу немає в топ-5: це приклад словникового розриву)")
    print(f"\nM2. Контекст за бюджету B = {B} слів")
    show_ctx("а) Усічення топ-5 документів (базова лінія)", truncate(top, B), answer)
    show_ctx("б) Відбір речень (метод)", select(text, top, prior, B), answer)


def pick_example(rng, B):
    """Випадкове тестове запитання, де усічення втрачає відповідь, а відбір речень зберігає."""
    for qi in rng.permutation(test_ans)[:400]:
        top, _, prior = retrieve(q[qi])
        if gold[qi] not in top:
            continue
        a = ans[qi].lower()
        if a not in " ".join(s for _, s in truncate(top, B)).lower() and \
           a in " ".join(s for _, s in select(q[qi], top, prior, B)).lower():
            return qi
    return int(rng.choice(test_ans))


def main():
    print(__doc__.split("Запуск")[0])
    rng = np.random.default_rng(int(os.environ.get("DEMO_SEED", 7)))
    B = 40
    while True:
        try:
            cmd = input("\n> Enter = приклад | текст запитання | b <слів> | q: ").strip()
        except EOFError:
            break
        if cmd in ("q", "quit", "exit"):
            break
        if cmd.startswith("b ") and cmd[2:].strip().isdigit():
            B = int(cmd[2:]); print(f"Бюджет змінено на {B} слів"); continue
        if cmd == "":
            qi = pick_example(rng, B)
            run(q[qi], B, ans[qi], gold[qi])
        else:
            run(cmd, B)


if __name__ == "__main__":
    main()
