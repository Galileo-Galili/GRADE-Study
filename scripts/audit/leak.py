import os, sys, csv, io, collections, hashlib, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
csv.field_size_limit(10**9)

BASE = r'c:\Users\user\Documents\Education Domain'
d = os.path.join(BASE, 'archive', 'my_dataset_expanded')

texts = {'human': [], 'ai': []}
for cls in ['human', 'ai']:
    root = os.path.join(d, cls)
    for cell in os.listdir(root):
        cd = os.path.join(root, cell)
        for fn in os.listdir(cd):
            texts[cls].append(open(os.path.join(cd, fn), encoding='utf-8', errors='replace').read())

print('== [AI-Generated] marker prefix rate ==')
for cls in ['human', 'ai']:
    n = sum(1 for t in texts[cls] if t.lstrip().startswith('[AI-Generated]'))
    any_ = sum(1 for t in texts[cls] if '[AI-Generated]' in t)
    print(f'  {cls}: startswith {n}/{len(texts[cls])} ({100*n/len(texts[cls]):.1f}%)  contains-anywhere {any_}')

# strip the marker and see if classes are otherwise distinguishable / duplicated
def strip(t):
    return re.sub(r'\[AI-Generated\]\s*', '', t).strip()

hh = collections.Counter(hashlib.md5(strip(t).encode('utf8', 'replace')).hexdigest() for t in texts['human'])
ah = collections.Counter(hashlib.md5(strip(t).encode('utf8', 'replace')).hexdigest() for t in texts['ai'])
overlap = set(hh) & set(ah)
print(f'\n== after stripping the marker ==')
print(f'  distinct human texts: {len(hh)} / {len(texts["human"])}')
print(f'  distinct ai texts:    {len(ah)} / {len(texts["ai"])}')
print(f'  EXACT texts appearing in BOTH classes: {len(overlap)}')

# how many unique "sentence templates" overall
sents = collections.Counter()
for cls in texts:
    for t in texts[cls]:
        for s in re.split(r'(?<=[.!?])\s+', strip(t)):
            s = s.strip()
            if len(s) > 30:
                sents[s] += 1
print(f'\n== sentence reuse ==')
print(f'  distinct sentences (>30 chars) across all 10240 docs: {len(sents)}')
print('  top 5 most repeated:')
for s, n in sents.most_common(5):
    print(f'    x{n}: {s[:110]}')
