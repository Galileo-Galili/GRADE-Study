"""Can the 9,481 topic-unknown essays be assigned to one of the 15 known topics?

Method. Fit a TF-IDF classifier on the topic-labelled essays, predicting the
topic rather than authorship. Check it against held-out labelled essays first:
if it cannot recover a known topic reliably, its guesses about the unknown
essays are worthless. Then apply it to the unknowns and look at how confident
it is.

An assignment is only kept where the classifier is confident. Anything below
the threshold stays unknown, because guessing is what corrupted the labels the
first time.
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(REPO, 'datasets', 'merged', 'essays_merged.csv')
OUT = os.path.join(REPO, 'datasets', 'merged', 'unknown_topic_guesses.csv')
SEED = 42
CONF = 0.60          # keep an assignment only above this probability


def main():
    m = pd.read_csv(MERGED)
    known = m[m['topic'].notna()].copy()
    unknown = m[m['topic'].isna()].copy()
    print('labelled essays : %d' % len(known))
    print('unknown essays  : %d  (%d AI / %d student)'
          % (len(unknown), (unknown.label == 1).sum(), (unknown.label == 0).sum()))
    print()

    # ---- can we recover a topic at all? test on held-out labelled essays
    Xtr, Xte, ytr, yte = train_test_split(
        known['text'].astype(str), known['topic'],
        test_size=0.2, random_state=SEED, stratify=known['topic'])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000,
                          sublinear_tf=True)
    Ztr = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=3000, C=5.0)
    clf.fit(Ztr, ytr)

    Zte = vec.transform(Xte)
    pred = clf.predict(Zte)
    prob = clf.predict_proba(Zte).max(1)
    acc = accuracy_score(yte, pred)
    print('=== sanity check on held-out LABELLED essays ===')
    print('  topic accuracy, all           : %.3f' % acc)
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        k = prob >= t
        if k.sum():
            print('  topic accuracy, conf >= %.1f   : %.3f   (keeps %.0f%%)'
                  % (t, accuracy_score(yte[k], pred[k]), 100 * k.mean()))
    print()

    # ---- apply to the unknowns
    Zu = vec.transform(unknown['text'].astype(str))
    up = clf.predict(Zu)
    uprob = clf.predict_proba(Zu).max(1)
    unknown['topic_guess'] = up
    unknown['confidence'] = uprob

    print('=== the unknown essays ===')
    print('  mean confidence : %.3f  (labelled held-out: %.3f)'
          % (uprob.mean(), prob.mean()))
    print('  above %.2f       : %d of %d (%.0f%%)'
          % (CONF, (uprob >= CONF).sum(), len(unknown), 100 * (uprob >= CONF).mean()))
    print()
    conf = unknown[unknown['confidence'] >= CONF]
    print('  confident guesses by topic:')
    ct = pd.crosstab(conf['topic_guess'], conf['label'])
    ct.columns = [c for c in ['student', 'AI'][:ct.shape[1]]]
    ct['total'] = ct.sum(1)
    print(ct.sort_values('total', ascending=False).to_string())

    unknown[['uid', 'label', 'topic_guess', 'confidence']].to_csv(OUT, index=False)
    print()
    print('wrote %s' % OUT)


if __name__ == '__main__':
    main()
