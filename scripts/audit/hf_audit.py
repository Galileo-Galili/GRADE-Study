import sys, io, csv, re, hashlib, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
csv.field_size_limit(10**9)

P = r'c:\Users\user\Documents\Education Domain\datasets\hf_pile\hf_pile_sample_20k.csv'
with open(P, encoding='utf-8', errors='replace', newline='') as f:
    rows = list(csv.DictReader(f))
print('rows', len(rows), dict(collections.Counter(r['source'] for r in rows)))


def norm(t):
    return re.sub(r'[^a-z0-9 ]', '', re.sub(r'\s+', ' ', t.lower())).strip()


# 1. marker leakage: any class-exclusive giveaway prefix/token?
print('\n=== 1. obvious marker leakage ===')
for cls in ['human', 'ai']:
    sub = [r['text'] for r in rows if r['source'] == cls]
    pref = collections.Counter(t.lstrip()[:24] for t in sub)
    print(f'  {cls}: top opening strings')
    for s, n in pref.most_common(3):
        print(f'      x{n:<5} {s!r}')

# 2. cross-class duplication
print('\n=== 2. exact cross-class duplication ===')
H = {hashlib.md5(norm(r['text']).encode()).hexdigest() for r in rows if r['source'] == 'human'}
A = {hashlib.md5(norm(r['text']).encode()).hexdigest() for r in rows if r['source'] == 'ai'}
print(f'  distinct human {len(H)} / distinct ai {len(A)} / shared {len(H & A)}')

# 3. within-class duplication
print('\n=== 3. within-class duplication ===')
for cls, S in [('human', 'human'), ('ai', 'ai')]:
    sub = [norm(r['text']) for r in rows if r['source'] == cls]
    print(f'  {cls}: {len(sub)} rows -> {len(set(sub))} distinct')

# 4. sentence-pool collapse (the AI_vs_Human_v2 failure mode)
print('\n=== 4. sentence diversity ===')
for cls in ['human', 'ai']:
    sents = collections.Counter()
    for r in rows:
        if r['source'] == cls:
            for s in re.split(r'(?<=[.!?])\s+', r['text']):
                s = s.strip()
                if len(s) > 30:
                    sents[s] += 1
    tot = sum(sents.values())
    print(f'  {cls}: {len(sents)} distinct sentences / {tot} total')
    for s, n in sents.most_common(2):
        print(f'      x{n:<4} {s[:80]!r}')

# 5. length confound - can length alone separate classes?
print('\n=== 5. length confound ===')
for cls in ['human', 'ai']:
    L = sorted(len(r['text']) for r in rows if r['source'] == cls)
    print(f'  {cls}: p05={L[int(len(L)*.05)]} p50={L[len(L)//2]} p95={L[int(len(L)*.95)]} max={L[-1]}')

# 6. trivial baseline - if this is near 1.0, something leaks
print('\n=== 6. TF-IDF baseline (chance=0.5) ===')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = [r['text'] for r in rows]
y = [1 if r['source'] == 'ai' else 0 for r in rows]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)
m = LogisticRegression(max_iter=2000).fit(v.fit_transform(Xtr), ytr)
print(f'  word 1-2gram: {accuracy_score(yte, m.predict(v.transform(Xte))):.4f}')

# length-only baseline
import numpy as np
Ltr = np.array([[len(t)] for t in Xtr]); Lte = np.array([[len(t)] for t in Xte])
m2 = LogisticRegression(max_iter=2000).fit(Ltr, ytr)
print(f'  LENGTH ONLY : {accuracy_score(yte, m2.predict(Lte)):.4f}')

print('\n=== samples ===')
for cls in ['human', 'ai']:
    for r in [x for x in rows if x['source'] == cls][:2]:
        print(f'\n--- {cls} (len {len(r["text"])}) ---')
        print(repr(r['text'][:300]))
