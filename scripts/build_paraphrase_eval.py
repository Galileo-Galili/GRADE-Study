"""Build per-generator paraphrase evaluation sets from the GLC-AIText corpus.

Purpose. The paraphrase finding in this study currently rests on 304 academic
abstracts, where both the register and the generating task differ from the
training corpus. This corpus isolates the generating task: every AI document
is a paraphrase of a specific human document, and both carry the same uid, so
the content is held fixed and only authorship changes.

Three generators paraphrase the same source articles, and they differ sharply
in output length relative to their human sources:

    gpt-3.5   1.9 : 1
    claude    4.1 : 1
    llama     5.8 : 1

That spread is itself a control. A failure that is uniform across the three is
not driven by length; a failure that tracks the ratio is.

Writes one balanced CSV per generator to data/, plus a leakage report against
the training corpus.
"""

import hashlib
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'Previous Logo Work', 'all_samples.csv')
DATA = os.path.join(REPO, 'data')
MIN_CHARS = 100
SEED = 42


def norm(s):
    return hashlib.md5(str(s).strip().lower().encode('utf8', 'ignore')).hexdigest()


def main():
    d = pd.read_csv(SRC)
    print('source rows: %d' % len(d))

    d['text'] = d['text'].astype(str)
    d = d[d['text'].str.len() >= MIN_CHARS]
    print('after min-length filter: %d' % len(d))

    human = d[d['label'] == 0][['uid', 'text']].drop_duplicates('uid')
    human_by_uid = human.set_index('uid')['text']

    rows = []
    for gen in ['gpt-3.5', 'claude', 'llama']:
        ai = d[d['generator'] == gen][['uid', 'text']].drop_duplicates('uid')
        # keep only paraphrases whose own source article is present
        ai = ai[ai['uid'].isin(human_by_uid.index)]
        n = len(ai)

        pairs = pd.DataFrame({
            'uid': list(ai['uid']) + list(ai['uid']),
            'text': list(ai['text']) + list(human_by_uid.loc[ai['uid']]),
            'label': [1] * n + [0] * n,
        })

        out = os.path.join(DATA, 'eval_para_%s.csv' % gen.replace('.', ''))
        pairs.to_csv(out, index=False)

        ai_med = ai['text'].str.len().median()
        hu_med = human_by_uid.loc[ai['uid']].str.len().median()
        print('%-9s pairs=%5d  rows=%5d  median chars AI=%5.0f human=%5.0f  ratio=%.1f:1'
              % (gen, n, 2 * n, ai_med, hu_med, hu_med / ai_med))
        rows.append({'generator': gen, 'pairs': n, 'rows': 2 * n,
                     'median_ai': ai_med, 'median_human': hu_med,
                     'ratio': hu_med / ai_med})

    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'paraphrase_stats.csv'), index=False)

    # ---- leakage audit against the training corpus, as the protocol requires
    print()
    train = pd.read_csv(os.path.join(DATA, 'split_train.csv'))
    tr_hashes = set(train['text'].map(norm))
    for gen in ['gpt-35', 'claude', 'llama']:
        ev = pd.read_csv(os.path.join(DATA, 'eval_para_%s.csv' % gen))
        overlap = len(set(ev['text'].map(norm)) & tr_hashes)
        print('leakage %-8s vs training corpus: %d' % (gen, overlap))
        assert overlap == 0, 'LEAK: %s shares %d documents with training' % (gen, overlap)
    print('\nall slices disjoint from training')


if __name__ == '__main__':
    main()
