import sys, csv, io, collections, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
csv.field_size_limit(10**9)

p = r'c:\Users\user\Documents\Education Domain\audit_evidence\AI_vs_Human_Text_Dataset_v2.csv'
with open(p, encoding='utf-8', errors='replace', newline='') as f:
    rows = list(csv.DictReader(f))
print('rows:', len(rows))
print('label:', dict(collections.Counter(r['label'] for r in rows)))
print('label_binary:', dict(collections.Counter(r['label_binary'] for r in rows)))
print('writing_style:', dict(collections.Counter(r['writing_style'] for r in rows)))
print('topic:', dict(collections.Counter(r['topic'] for r in rows)))

# is label_binary a clean function of label?
pair = collections.Counter((r['label'], r['label_binary']) for r in rows)
print('label x label_binary:', dict(pair))

# LEAKAGE: do the engineered numeric features trivially separate the classes?
print('\n== ai_probability_score / detection_confidence by class ==')
for lab in sorted(set(r['label'] for r in rows)):
    sub = [r for r in rows if r['label'] == lab]
    for col in ['ai_probability_score', 'detection_confidence', 'text_length']:
        vals = [float(r[col]) for r in sub if r[col]]
        vals.sort()
        print(f'  {lab:6s} {col:22s} min={vals[0]:.3f} med={vals[len(vals)//2]:.3f} max={vals[-1]:.3f}')

# text duplication and sentence reuse
texts = [r['text_content'] for r in rows]
print('\ndistinct text_content:', len(set(texts)), '/', len(texts))
sents = collections.Counter()
for t in texts:
    for s in re.split(r'(?<=[.!?])\s+', t):
        s = s.strip()
        if len(s) > 30:
            sents[s] += 1
print('distinct sentences (>30 chars):', len(sents))
print('top 5 repeated:')
for s, n in sents.most_common(5):
    print(f'  x{n}: {s[:110]}')

print('\n== full sample docs ==')
for lab in sorted(set(r['label'] for r in rows)):
    ex = next(r for r in rows if r['label'] == lab)
    print(f'\n--- label={lab} ---')
    print(ex['text_content'][:700])
