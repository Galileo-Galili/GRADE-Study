import sys, io, collections, random, os, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyarrow.parquet as pq
import fsspec

BASE = 'https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/'
HUMAN_SHARDS = [
    'data/train-00000-of-00007-bc5952582e004d67.parquet',
    'data/train-00002-of-00007-ee2d43f396e78fbc.parquet',
    'data/train-00004-of-00007-b269dc49374a2c0b.parquet',
]
AI_SHARDS = [
    'data/train-00006-of-00007-3d8a471ba0cf1c8d.parquet',
    'data/train-00005-of-00007-3dce5e05ddbad789.parquet',
]
TARGET_PER_CLASS = 10000
rng = random.Random(42)
fs = fsspec.filesystem('http')

rows = []


def harvest(shards, want_src, need):
    got = 0
    for fn in shards:
        if got >= need:
            break
        f = pq.ParquetFile(fs.open(BASE + fn))
        ng = f.metadata.num_row_groups
        # sample row groups spread across the shard for diversity
        idxs = list(range(ng))
        rng.shuffle(idxs)
        for gi in idxs:
            if got >= need:
                break
            t = f.read_row_group(gi, columns=['source', 'id', 'text'])
            src = t.column('source').to_pylist()
            ids = t.column('id').to_pylist()
            txt = t.column('text').to_pylist()
            for s, i, x in zip(src, ids, txt):
                if s == want_src and x and got < need:
                    rows.append({'source': s, 'id': i, 'text': x, 'shard': fn.split('/')[-1][:22]})
                    got += 1
        print(f'  after {fn.split("/")[-1][:28]}: {got}/{need} {want_src}')
    return got


print('harvesting human...')
harvest(HUMAN_SHARDS, 'human', TARGET_PER_CLASS)
print('harvesting ai...')
harvest(AI_SHARDS, 'ai', TARGET_PER_CLASS)

print('\ntotal:', len(rows), dict(collections.Counter(r['source'] for r in rows)))

out = r'c:\Users\user\Documents\Education Domain\datasets\hf_pile\hf_pile_sample_20k.csv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['source', 'id', 'shard', 'text'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('wrote', out, f'{os.path.getsize(out)/1e6:.1f} MB')
