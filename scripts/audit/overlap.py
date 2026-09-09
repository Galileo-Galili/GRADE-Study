import sys, csv, io, re, hashlib, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
csv.field_size_limit(10**9)

BASE = r'c:\Users\user\Documents\Education Domain'

def load(p, tcol, lcol):
    with open(p, encoding='utf-8', errors='replace', newline='') as f:
        return [(r[tcol], r[lcol]) for r in csv.DictReader(f)]

aide = load(BASE + r'\AIDE\AIDE_train_essays.csv', 'text', 'generated')
llm = load(BASE + r'\datasets\llm_detect\Training_Essay_Data.csv', 'text', 'generated')

def norm(t):
    t = t.lower()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^a-z0-9 ]', '', t)
    return t.strip()

def h(t):
    return hashlib.md5(norm(t).encode()).hexdigest()

ah = {h(t): l for t, l in aide}
lh = collections.defaultdict(list)
for t, l in llm:
    lh[h(t)].append(l)

exact = set(ah) & set(lh)
print('=== EXACT normalized-text overlap ===')
print(f'  AIDE distinct:       {len(ah)}')
print(f'  LLM-Detect distinct: {len(lh)}')
print(f'  overlap:             {len(exact)}')
print(f'  = {100*len(exact)/len(ah):.1f}% of AIDE appears in LLM-Detect')

# label agreement on overlapping docs
dis = [(k, ah[k], set(lh[k])) for k in exact if {ah[k]} != set(lh[k])]
print(f'  label disagreements on shared docs: {len(dis)}')

# internal duplication in LLM-Detect (paper reported 27,340 but file has 29,145)
print('\n=== LLM-Detect internal duplication ===')
dup = {k: v for k, v in lh.items() if len(v) > 1}
print(f'  texts appearing >1x: {len(dup)}  (extra rows: {sum(len(v)-1 for v in dup.values())})')
crossdup = {k: v for k, v in dup.items() if len(set(v)) > 1}
print(f'  duplicated texts with CONFLICTING labels: {len(crossdup)}')

# 8-gram containment: is AIDE human text reused inside LLM-Detect?
def grams(t, n=8):
    w = norm(t).split()
    return {' '.join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}

llm_grams = set()
for t, l in llm:
    llm_grams |= grams(t)
print('\n=== near-duplicate check (8-gram containment) ===')
hits = 0
sample = [t for t, l in aide if l == '0'][:400]
for t in sample:
    g = grams(t)
    if g and len(g & llm_grams) / len(g) > 0.8:
        hits += 1
print(f'  AIDE human essays with >80% of 8-grams present in LLM-Detect: {hits}/{len(sample)}')
