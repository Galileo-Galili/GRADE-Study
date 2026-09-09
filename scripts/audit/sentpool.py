import sys, csv, io, collections, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
csv.field_size_limit(10**9)

p = r'c:\Users\user\Documents\Education Domain\audit_evidence\AI_vs_Human_Text_Dataset_v2.csv'
with open(p, encoding='utf-8', errors='replace', newline='') as f:
    rows = list(csv.DictReader(f))

def sset(t):
    return {s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if len(s.strip()) > 30}

H, A = set(), set()
for r in rows:
    (H if r['label'] == 'Human' else A).add_all if False else None
for r in rows:
    tgt = H if r['label'] == 'Human' else A
    tgt |= sset(r['text_content'])

print('distinct sentences in Human rows:', len(H))
print('distinct sentences in AI rows:   ', len(A))
print('shared by BOTH classes:          ', len(H & A))
print('exclusive to Human:              ', len(H - A))
print('exclusive to AI:                 ', len(A - H))

# Can a bag-of-sentences even separate them? Try a trivial holdout with logistic regression on TF-IDF.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = [r['text_content'] for r in rows]
    y = [1 if r['label'] == 'AI' else 0 for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    m = LogisticRegression(max_iter=2000).fit(v.fit_transform(Xtr), ytr)
    acc = accuracy_score(yte, m.predict(v.transform(Xte)))
    print(f'\nTF-IDF + LogReg holdout accuracy: {acc:.4f}  (chance = 0.5000)')
except ImportError:
    print('\n(sklearn unavailable - skipped baseline)')
