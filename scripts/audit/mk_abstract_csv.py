"""Flatten the 304 Abstract-Paraphrase .txt files into one CSV.

`pair_id` is the shared source-abstract identifier (e.g. d10182 from
G-d10182Abstract.txt / N-d10182Abstract.txt). It MUST be used as the
grouping key at split time: each human abstract and its GPT-4 paraphrase
share a pair_id, and separating them across splits would place near
duplicate text on both sides.
"""
import os, csv, re, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'c:\Users\user\Documents\Education Domain\datasets\abstract_paraphrase'
SRC = os.path.join(BASE, 'EnglishDatasetSample')
OUT = os.path.join(BASE, 'abstract_paraphrase_en.csv')

rows = []
for cls, sub, label in [('Native', 'Native/arh-English', 0),
                        ('GPT', 'GPT/arh-English', 1)]:
    d = os.path.join(SRC, *sub.split('/'))
    for fn in sorted(os.listdir(d)):
        text = open(os.path.join(d, fn), encoding='utf-8', errors='replace').read().strip()
        m = re.search(r'(d\d+)', fn)
        pair_id = m.group(1) if m else fn
        rows.append({'pair_id': pair_id, 'source': cls, 'label': label,
                     'filename': fn, 'text': text})

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['pair_id', 'source', 'label', 'filename', 'text'])
    w.writeheader()
    w.writerows(rows)

print('rows:', len(rows))
print('label counts:', dict(collections.Counter(r['label'] for r in rows)))
pc = collections.Counter(r['pair_id'] for r in rows)
print('distinct pair_id:', len(pc))
print('pair_ids with both members:', sum(1 for v in pc.values() if v == 2))
print('pair_ids NOT paired:', {k: v for k, v in pc.items() if v != 2} or 'none')
L = sorted(len(r['text']) for r in rows)
print(f'text length: min={L[0]} med={L[len(L)//2]} max={L[-1]}')
print('wrote', OUT, f'{os.path.getsize(OUT)/1e3:.1f} KB')
