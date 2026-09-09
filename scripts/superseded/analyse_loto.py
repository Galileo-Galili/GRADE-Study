"""Turn the leave-one-assignment-out results into the paper's two tables.

    python scripts/analyse_loto.py

Three parts.

  Controls. The length-only and word-frequency comparisons, fitted
  separately on each fold's test set. CPU only, cached to
  results/loto_baselines.csv, so this part can be run while the GPU sweep
  is still going.

  RQ1. Per-fold accuracy, recall and ROC-AUC against those controls, with
  the mean and the worst fold. Emits results/table_loto.tex.

  RQ2. The false-positive rate on authentic work, attributed to the writer
  using PERSUADE's grades and characteristics. Joined on essay uid, not on
  row order. Emits results/table_fairness.tex and results/loto_fairness.csv.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOTO = os.path.join(REPO, 'data', 'loto')
RESULTS = os.path.join(REPO, 'results')
PREDS = os.path.join(RESULTS, 'loto_predictions')
RUNS = os.path.join(RESULTS, 'loto_runs.csv')
BASE = os.path.join(RESULTS, 'loto_baselines.csv')
PERSUADE = os.path.join(REPO, 'datasets', 'Persuade 2.0',
                        'persuade_2.0_human_scores_demo_id_github.csv')
SEED = 42
MIN_N = 30


def h(s):
    return hashlib.md5(str(s).strip().lower().encode('utf8', 'ignore')).hexdigest()


# ------------------------------------------------------------------ controls
def controls_for(df):
    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=SEED, stratify=y)
    out = {}

    Ltr = np.array([[len(t)] for t in Xtr])
    Lte = np.array([[len(t)] for t in Xte])
    m = LogisticRegression(max_iter=2000).fit(Ltr, ytr)
    out['length_only'] = (accuracy_score(yte, m.predict(Lte)),
                          recall_score(yte, m.predict(Lte)),
                          roc_auc_score(yte, m.predict_proba(Lte)[:, 1]))

    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)
    m = LogisticRegression(max_iter=2000).fit(v.fit_transform(Xtr), ytr)
    Z = v.transform(Xte)
    out['tfidf'] = (accuracy_score(yte, m.predict(Z)),
                    recall_score(yte, m.predict(Z)),
                    roc_auc_score(yte, m.predict_proba(Z)[:, 1]))
    return out


def build_controls(folds):
    if os.path.exists(BASE):
        return pd.read_csv(BASE)
    rows = []
    for r in folds.itertuples():
        te = pd.read_csv(os.path.join(LOTO, 'fold_%02d_test.csv' % r.fold))
        for name, (a, rc, au) in controls_for(te).items():
            rows.append({'fold': r.fold, 'held_out_topic': r.held_out_topic,
                         'baseline': name, 'accuracy': a, 'recall': rc,
                         'roc_auc': au, 'n_test': len(te)})
        print('  controls fold %02d done' % r.fold, flush=True)
    b = pd.DataFrame(rows)
    b.to_csv(BASE, index=False)
    return b


# ------------------------------------------------------------------ RQ1
def rq1(folds, runs, base):
    seed42 = runs[runs.seed == 42].set_index('fold')
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{Leave-one-assignment-out. Each row trains on the other 14',
             r'assignments and marks only the one named. Generated text is the',
             r'positive class; all figures are percentages. Controls are fitted',
             r'separately on each fold and cannot recognise authorship.}',
             r'\label{tab:loto}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             (r'\textbf{Held-out assignment} & \textbf{Test $n$} & '
              r'\textbf{Acc.} & \textbf{Recall} & \textbf{ROC} & '
              r'\textbf{vs.\ length} \\'), r'\midrule']

    lo = base[base.baseline == 'length_only'].set_index('fold')['accuracy']
    rows = []
    for r in folds.itertuples():
        if r.fold not in seed42.index:
            continue
        s = seed42.loc[r.fold]
        margin = 100 * (s['accuracy'] - lo.get(r.fold, np.nan))
        name = str(r.held_out_topic).replace('"', "``", 1)
        lines.append('%s & %s & %.2f & %.2f & %.2f & %+.1f \\\\'
                     % (name, '{:,}'.format(int(r.n_test)).replace(',', '{,}'),
                        100 * s['accuracy'], 100 * s['recall'],
                        100 * s['roc_auc'], margin))
        rows.append({'fold': r.fold, 'topic': r.held_out_topic,
                     'accuracy': s['accuracy'], 'recall': s['recall'],
                     'roc_auc': s['roc_auc'], 'length_only': lo.get(r.fold),
                     'margin_pts': margin})

    if not rows:
        print('no seed-42 runs yet; skipping RQ1 table')
        return None
    df = pd.DataFrame(rows)
    lines += [r'\midrule',
              r'\textbf{Mean over folds} & & %.2f & %.2f & %.2f & %+.1f \\'
              % (100 * df.accuracy.mean(), 100 * df.recall.mean(),
                 100 * df.roc_auc.mean(), df.margin_pts.mean()),
              r'\textbf{Worst fold} & & %.2f & %.2f & %.2f & %+.1f \\'
              % (100 * df.accuracy.min(), 100 * df.recall.min(),
                 100 * df.roc_auc.min(), df.margin_pts.min()),
              r'\midrule',
              r'\emph{word-frequency control, mean} & & %.2f & %.2f & %.2f & \\'
              % (100 * base[base.baseline == 'tfidf'].accuracy.mean(),
                 100 * base[base.baseline == 'tfidf'].recall.mean(),
                 100 * base[base.baseline == 'tfidf'].roc_auc.mean()),
              r'\emph{length-only control, mean} & & %.2f & %.2f & %.2f & \\'
              % (100 * base[base.baseline == 'length_only'].accuracy.mean(),
                 100 * base[base.baseline == 'length_only'].recall.mean(),
                 100 * base[base.baseline == 'length_only'].roc_auc.mean()),
              r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(RESULTS, 'table_loto.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    df.to_csv(os.path.join(RESULTS, 'loto_summary.csv'), index=False)

    print('=== RQ1: transfer to an unseen assignment ===')
    print(df.assign(accuracy=(100 * df.accuracy).round(2),
                    recall=(100 * df.recall).round(2),
                    roc_auc=(100 * df.roc_auc).round(2),
                    length_only=(100 * df.length_only).round(2),
                    margin_pts=df.margin_pts.round(1))
          [['topic', 'accuracy', 'recall', 'roc_auc', 'length_only', 'margin_pts']]
          .to_string(index=False))
    print()
    print('spread in accuracy across folds: %.2f points'
          % (100 * (df.accuracy.max() - df.accuracy.min())))

    # stage-2 candidates
    print('worst fold: %d (%s)   best fold: %d (%s)'
          % (df.loc[df.accuracy.idxmin(), 'fold'], df.loc[df.accuracy.idxmin(), 'topic'],
             df.loc[df.accuracy.idxmax(), 'fold'], df.loc[df.accuracy.idxmax(), 'topic']))
    return df


# ------------------------------------------------------------------ RQ2
GROUPS = [('holistic_essay_score', 'holistic grade'),
          ('ell_status', 'English an additional language'),
          ('grade_level', 'school year'),
          ('economically_disadvantaged', 'economic disadvantage')]


def rq2(folds, runs):
    p = pd.read_csv(PERSUADE, low_memory=False)
    p['uid'] = p['full_text'].map(h)
    p = p.drop_duplicates('uid').set_index('uid')

    frames = []
    for r in runs[runs.seed == 42].itertuples():
        f = os.path.join(PREDS, 'preds_f%02d_%s_seed%d.csv'
                         % (r.fold, r.model, r.seed))
        if not os.path.exists(f):
            continue
        pr = pd.read_csv(f)
        pr['fold'] = r.fold
        frames.append(pr)
    if not frames:
        print('no predictions yet; skipping RQ2')
        return None

    allp = pd.concat(frames, ignore_index=True)
    hu = allp[allp['label'] == 0].join(p[[g for g, _ in GROUPS]], on='uid',
                                       how='inner')
    print()
    print('=== RQ2: who is falsely accused ===')
    print('authentic essays with writer information: %d of %d (%.0f%%)'
          % (len(hu), (allp.label == 0).sum(),
             100 * len(hu) / max((allp.label == 0).sum(), 1)))
    print()

    rows = []
    for col, label in GROUPS:
        sub = hu.dropna(subset=[col])
        for val, grp in sub.groupby(col):
            if len(grp) < MIN_N:
                continue
            rows.append({'group': label, 'value': str(val), 'n': len(grp),
                         'false_positive_rate': grp['pred'].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, 'loto_fairness.csv'), index=False)

    for label in out['group'].unique():
        s = out[out['group'] == label]
        print('--- by %s ---' % label)
        print(s.assign(FPR_pct=(100 * s.false_positive_rate).round(2))
              [['value', 'n', 'FPR_pct']].to_string(index=False))
        print()

    # ---- LaTeX
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{The false-positive rate on authentic student work,',
             r'pooled over the 15 held-out assignments and broken down by the',
             r'grade the essay received and by whether English is an additional',
             r'language for the writer. Every figure is an accusation against a',
             r'student who did their own work.}',
             r'\label{tab:fairness}',
             r'\begin{tabular}{@{}lrr@{}}', r'\toprule',
             (r'\textbf{Writer group} & \textbf{Essays} & '
              r'\textbf{False-positive rate} \\'), r'\midrule']
    for col, label in GROUPS[:2]:
        s = out[out['group'] == label]
        if s.empty:
            continue
        lines.append(r'\multicolumn{3}{@{}l}{\emph{by %s}} \\' % label)
        for r in s.itertuples():
            lines.append(r'\quad %s & %s & %.2f \\'
                         % (r.value,
                            '{:,}'.format(int(r.n)).replace(',', '{,}'),
                            100 * r.false_positive_rate))
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']
    with open(os.path.join(RESULTS, 'table_fairness.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return out


def main():
    folds = pd.read_csv(os.path.join(LOTO, 'folds.csv'))
    print('fitting controls (CPU; cached to %s)' % os.path.basename(BASE))
    base = build_controls(folds)
    print()

    if not os.path.exists(RUNS):
        print('no detector runs yet. Controls are ready; rerun after the sweep.')
        print(base.pivot_table(index='held_out_topic', columns='baseline',
                               values='accuracy').mul(100).round(2).to_string())
        return

    runs = pd.read_csv(RUNS)
    rq1(folds, runs, base)
    rq2(folds, runs)
    print()
    print('wrote results/table_loto.tex, results/table_fairness.tex')


if __name__ == '__main__':
    main()
