"""Build leave-one-topic-out splits from the merged corpus.

For each of the 15 topics: train on the other 14, mark on the held out one.
Every test decision therefore concerns an assignment absent from training,
which is the shift a programme meets when it sets new coursework.

Three properties are enforced rather than assumed.

  Topic isolation. No essay from the held-out topic appears in training or
  validation. Essays whose topic is unknown are excluded entirely, since any
  of them might belong to the held-out topic.

  No shared text. Train, validation and test are disjoint on exact text. The
  check raises rather than warns.

  Balance. Every split is down-sampled to equal numbers of student and
  generated essays, so chance accuracy is 50% throughout and folds are
  comparable.

Writes data/loto/fold_<id>_{train,val,test}.csv and data/loto/folds.csv.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(REPO, 'datasets', 'merged', 'essays_merged.csv')
OUTDIR = os.path.join(REPO, 'data', 'loto')
SEED = 42
VAL_FRAC = 0.15
MIN_TEST_PER_CLASS = 150


def norm(s):
    return hashlib.md5(str(s).strip().lower().encode('utf8', 'ignore')).hexdigest()


def balance(df, rng):
    """Down-sample the majority class so the split is 50/50."""
    n = df['label'].value_counts().min()
    return pd.concat([
        df[df.label == 0].sample(n, random_state=rng),
        df[df.label == 1].sample(n, random_state=rng),
    ]).sample(frac=1, random_state=rng).reset_index(drop=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    m = pd.read_csv(MERGED)

    # topic-unknown essays cannot be used: they might be the held-out topic
    m = m[m['topic'].notna()].copy()
    m['H'] = m['text'].map(norm)
    print('topic-labelled essays: %d' % len(m))

    topics = (m.groupby('topic')['label'].apply(lambda s: s.value_counts().min())
              .sort_values(ascending=False))
    usable = topics[topics >= MIN_TEST_PER_CLASS]
    print('topics with >=%d per class: %d of %d'
          % (MIN_TEST_PER_CLASS, len(usable), len(topics)))
    print()

    rows = []
    for fid, topic in enumerate(usable.index):
        rng = SEED + fid
        held = m[m['topic'] == topic]
        rest = m[m['topic'] != topic]

        test = balance(held, rng)

        # validation is carved out of the remaining topics, never the held-out one.
        # Split stratified on the label so both parts stay 50/50 after balancing.
        rest_b = balance(rest, rng)
        tr_idx, va_idx = train_test_split(
            np.arange(len(rest_b)), test_size=VAL_FRAC,
            random_state=rng, stratify=rest_b['label'])
        # a stratified split can be off by one; re-balance so each part is exact
        train = balance(rest_b.iloc[tr_idx], rng)
        val = balance(rest_b.iloc[va_idx], rng)

        # ---- enforced audits
        for a, b, name in [(train, val, 'train/val'),
                           (train, test, 'train/test'),
                           (val, test, 'val/test')]:
            shared = set(a['H']) & set(b['H'])
            assert not shared, ('LEAK in fold %d (%s): %d shared texts'
                                % (fid, name, len(shared)))
        for part, name in [(train, 'train'), (val, 'val')]:
            bad = (part['topic'] == topic).sum()
            assert bad == 0, ('TOPIC LEAK in fold %d: %d held-out-topic essays in %s'
                              % (fid, bad, name))
        for part, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            vc = part['label'].value_counts()
            assert vc.iloc[0] == vc.iloc[-1], 'fold %d %s not balanced' % (fid, name)

        cols = ['uid', 'text', 'label', 'topic', 'generator']
        for part, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            part[cols].to_csv(
                os.path.join(OUTDIR, 'fold_%02d_%s.csv' % (fid, name)), index=False)

        rows.append({'fold': fid, 'held_out_topic': topic,
                     'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
                     'test_per_class': len(test) // 2})
        print('fold %02d  %-40s train=%5d val=%4d test=%5d'
              % (fid, topic[:40], len(train), len(val), len(test)))

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(OUTDIR, 'folds.csv'), index=False)
    print()
    print('all %d folds pass topic isolation, text disjointness and balance'
          % len(folds))
    print('wrote %s' % OUTDIR)


if __name__ == '__main__':
    main()
