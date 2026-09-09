"""Who gets falsely accused?

A false positive is an authentic student essay flagged as generated. Aggregate
accuracy hides who absorbs those errors. PERSUADE 2.0 carries per-writer
demographics for the same essays our detectors were marked on, so the errors
can be attributed: English-language-learner status, holistic essay score,
grade level, economic disadvantage, disability status.

Uses the predictions already written by the sweep. No training, no GPU.
Writes results/fairness.csv.
"""

import hashlib
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
PREDS = os.path.join(REPO, 'results', 'predictions')
OUT = os.path.join(REPO, 'results', 'fairness.csv')

PERSUADE = os.path.join(REPO, 'datasets', 'Persuade 2.0',
                        'persuade_2.0_human_scores_demo_id_github.csv')

MODELS = ['bert', 'roberta', 'deberta-v3', 'distilbert', 'electra']
EVALS = [('pooled', 'split_test', 'indist'),
         ('xprompt_p0to1', 'xprompt_p0to1_test', 'other_assignment'),
         ('xprompt_p1to0', 'xprompt_p1to0_test', 'other_assignment')]

GROUPS = ['ell_status', 'holistic_essay_score', 'grade_level',
          'economically_disadvantaged', 'student_disability_status']
MIN_N = 30


def h(s):
    return hashlib.md5(str(s).strip().lower().encode('utf8', 'ignore')).hexdigest()


def main():
    p = pd.read_csv(PERSUADE, low_memory=False)
    p['H'] = p['full_text'].map(h)
    p = p.drop_duplicates('H')
    meta = p.set_index('H')[GROUPS]

    rows = []
    for cond, split, es in EVALS:
        ev = pd.read_csv(os.path.join(DATA, '%s.csv' % split)).reset_index(drop=True)
        ev['H'] = ev['text'].map(h)

        for m in MODELS:
            f = os.path.join(PREDS, 'preds_%s_%s_seed42_%s.csv' % (cond, m, es))
            if not os.path.exists(f):
                continue
            pr = pd.read_csv(f)
            if len(pr) != len(ev):
                print('[skip] %s/%s length mismatch %d vs %d'
                      % (cond, m, len(pr), len(ev)))
                continue

            df = ev.copy()
            df['pred'] = pr['pred'].values
            df['label_chk'] = pr['label'].values
            assert (df['label'] == df['label_chk']).all(), 'row order mismatch'

            # false positives are only defined on authentic work
            hu = df[df['label'] == 0].join(meta, on='H', how='inner')

            for g in GROUPS:
                sub = hu.dropna(subset=[g])
                for val, grp in sub.groupby(g):
                    if len(grp) < MIN_N:
                        continue
                    rows.append({
                        'condition': cond, 'model': m, 'group': g,
                        'value': str(val), 'n': len(grp),
                        'false_positive_rate': grp['pred'].mean(),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print('wrote %s  (%d rows)\n' % (OUT, len(out)))

    # ---- headline: ELL gap, averaged over the five detectors
    for cond in out['condition'].unique():
        e = out[(out['condition'] == cond) & (out['group'] == 'ell_status')]
        if e.empty:
            continue
        piv = e.pivot_table(index='model', columns='value',
                            values='false_positive_rate')
        if {'Yes', 'No'} <= set(piv.columns):
            piv['gap_pts'] = 100 * (piv['Yes'] - piv['No'])
            print('=== %s : false-positive rate on authentic essays ===' % cond)
            print((100 * piv[['No', 'Yes']]).round(2)
                  .assign(gap_pts=piv['gap_pts'].round(2)).to_string())
            print()

    # ---- essay score
    s = out[out['group'] == 'holistic_essay_score']
    if not s.empty:
        print('=== false-positive rate by holistic essay score (1=weakest) ===')
        print((100 * s.pivot_table(index='condition', columns='value',
                                   values='false_positive_rate')).round(2).to_string())


if __name__ == '__main__':
    main()
