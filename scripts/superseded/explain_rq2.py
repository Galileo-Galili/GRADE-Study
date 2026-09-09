"""Why are highly graded essays flagged? The mechanism behind RQ2.

    python scripts/explain_rq2.py

RQ2 establishes that false accusations are distributed unevenly across
students. This script asks what the detector is responding to when it makes
them. It is a mechanism check for RQ2, not a third research question, and
nothing here is reported unless RQ2 itself shows an uneven distribution.

Two parts, deliberately of different kinds.

  Part A, all essays, no sampling. Counts formal discourse markers
  ("however", "furthermore", "moreover", ...) in every authentic essay and
  asks whether marker density predicts being flagged, and whether it does so
  beyond what the grade already explains. Deterministic and fully powered:
  it uses every authentic essay in every fold.

  Part B, a sample, with a model. LIME on authentic essays the detector
  flagged, contrasted with authentic essays it passed. Answers which words
  pushed a specific real essay toward "generated". Needs the kept checkpoint
  from scripts/train_loto.py.

If Part A shows marker density predicting flags independently of grade, and
Part B's flagged essays are driven by those same markers, the mechanism is
that the detector learned formal polish as evidence of generation. If Part A
is flat, the grade effect has some other cause and this script should say so
rather than reach.
"""

import hashlib
import os
import re
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
PREDS = os.path.join(RESULTS, 'loto_predictions')
LOTO = os.path.join(REPO, 'data', 'loto')
PERSUADE = os.path.join(REPO, 'datasets', 'Persuade 2.0',
                        'persuade_2.0_human_scores_demo_id_github.csv')

MODEL = 'deberta-v3'
LIME_FOLD = 0
N_EXPLAIN = 60
NUM_SAMPLES = 500
MAX_LEN, BATCH = 256, 16

# Formal connectives and transitions. Chosen before looking at any LIME
# output, from the discourse-marker literature rather than from our results,
# so Part A cannot be tuned to agree with Part B.
MARKERS = ['however', 'furthermore', 'moreover', 'additionally',
           'consequently', 'therefore', 'nevertheless', 'nonetheless',
           'in conclusion', 'in addition', 'for instance', 'as a result',
           'on the other hand', 'in contrast', 'ultimately', 'notably',
           'significantly', 'essentially', 'overall', 'thus']

STOP = set(("a an the and or but if while of to in on at by for with from as is "
            "are was were be been being it its this that these those there here "
            "we our you your they their he she his her not no than then so such "
            "can could would should may might will just also very more most "
            "other some any each which who whom what when where how why do does "
            "did done have has had i me my").split())


def h(s):
    return hashlib.md5(str(s).strip().lower().encode('utf8', 'ignore')).hexdigest()


def load_authentic():
    """Every authentic essay across all folds, with verdict and grade."""
    p = pd.read_csv(PERSUADE, low_memory=False)
    p['uid'] = p['full_text'].map(h)
    p = p.drop_duplicates('uid').set_index('uid')

    frames = []
    for f in sorted(os.listdir(PREDS)):
        if not f.endswith('.csv') or '_seed42' not in f:
            continue
        fold = int(re.search(r'_f(\d+)_', f).group(1))
        pr = pd.read_csv(os.path.join(PREDS, f))
        te = pd.read_csv(os.path.join(LOTO, 'fold_%02d_test.csv' % fold))
        pr = pr.merge(te[['uid', 'text', 'topic']], on='uid', how='left')
        pr['fold'] = fold
        frames.append(pr)
    if not frames:
        raise SystemExit('no seed-42 predictions in %s' % PREDS)

    allp = pd.concat(frames, ignore_index=True)
    hu = allp[allp['label'] == 0].copy()
    hu = hu.join(p[['holistic_essay_score', 'word_count']], on='uid', how='inner')
    return hu.dropna(subset=['text', 'holistic_essay_score'])


# ------------------------------------------------------------------ Part A
def marker_rate(text):
    t = ' ' + str(text).lower() + ' '
    words = max(len(t.split()), 1)
    hits = sum(t.count(' ' + m + ' ') if ' ' in m else
               len(re.findall(r'\b%s\b' % re.escape(m), t)) for m in MARKERS)
    return 1000.0 * hits / words          # markers per 1000 words


def part_a(hu):
    print('=' * 68)
    print('PART A  -  does formal-marker density predict a false accusation?')
    print('=' * 68)
    hu = hu.copy()
    hu['markers'] = hu['text'].map(marker_rate)
    hu['grade'] = hu['holistic_essay_score'].astype(int)

    print('authentic essays analysed: %d  (no sampling)' % len(hu))
    print()
    print('correlation with being flagged as generated')
    for c, lab in [('grade', 'essay grade'), ('markers', 'marker density'),
                   ('word_count', 'word count')]:
        print('   %-16s %+.3f' % (lab, hu[c].corr(hu['pred'])))
    print('   grade vs markers %+.3f' % hu['grade'].corr(hu['markers']))
    print()

    print('marker density and flag rate by grade')
    g = hu.groupby('grade').agg(n=('pred', 'size'),
                                flagged_pct=('pred', lambda s: 100 * s.mean()),
                                markers_per_1k=('markers', 'mean'))
    print(g.round(2).to_string())
    print()

    # The key test: within a grade band, does marker density still separate
    # flagged from unflagged? If yes, polish predicts flags beyond grade.
    print('within each grade, mean marker density of flagged vs passed essays')
    rows = []
    for grade, grp in hu.groupby('grade'):
        f = grp[grp.pred == 1]['markers']
        u = grp[grp.pred == 0]['markers']
        if len(f) < 15 or len(u) < 15:
            continue
        rows.append({'grade': grade, 'n_flagged': len(f), 'n_passed': len(u),
                     'markers_flagged': f.mean(), 'markers_passed': u.mean(),
                     'difference': f.mean() - u.mean()})
    if rows:
        w = pd.DataFrame(rows)
        print(w.round(2).to_string(index=False))
        print()
        pos = (w['difference'] > 0).sum()
        print('grades where flagged essays are more marker-dense: %d of %d'
              % (pos, len(w)))
        if pos == len(w) and len(w) >= 3:
            print('-> consistent: polish predicts a flag beyond what grade explains')
        elif pos <= len(w) / 2:
            print('-> NOT supported: marker density does not separate within grade.')
            print('   Do not claim polish as the mechanism; report this as open.')
        else:
            print('-> mixed; report as suggestive only')
        w.to_csv(os.path.join(RESULTS, 'rq2_markers_within_grade.csv'), index=False)
    else:
        print('too few flagged essays per grade to test within-grade')

    hu[['uid', 'fold', 'grade', 'markers', 'word_count', 'pred']].to_csv(
        os.path.join(RESULTS, 'rq2_marker_density.csv'), index=False)
    return hu


# ------------------------------------------------------------------ Part B
def part_b(hu):
    print()
    print('=' * 68)
    print('PART B  -  LIME on real false accusations')
    print('=' * 68)
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print('lime not installed:  pip install lime')
        return None

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cd = os.path.join(REPO, 'models', 'loto',
                      'f%02d_%s_seed42' % (LIME_FOLD, MODEL))
    if not os.path.isdir(cd):
        print('no checkpoint at %s' % cd)
        print('set KEEP_CHECKPOINT_FOLDS in train_loto.py and rerun that fold')
        return None

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(cd)
    model = AutoModelForSequenceClassification.from_pretrained(cd).to(dev).eval()

    def proba(texts):
        out = []
        for i in range(0, len(texts), BATCH):
            enc = tok(list(texts[i:i + BATCH]), max_length=MAX_LEN,
                      truncation=True, padding='max_length',
                      return_tensors='pt').to(dev)
            with torch.no_grad():
                out.append(torch.softmax(model(**enc).logits, 1).cpu().numpy())
        return np.vstack(out)

    sub = hu[hu.fold == LIME_FOLD]
    flagged = sub[(sub.pred == 1) & (sub.holistic_essay_score >= 5)]
    passed = sub[(sub.pred == 0) & (sub.holistic_essay_score <= 2)]
    print('fold %d: %d flagged high-grade essays, %d passed low-grade essays'
          % (LIME_FOLD, len(flagged), len(passed)))
    if len(flagged) < 10:
        print('too few high-grade false positives in this fold to explain.')
        print('Report Part A only.')
        return None

    explainer = LimeTextExplainer(class_names=['authentic', 'generated'])
    rows = []
    for name, grp in [('flagged_high_grade', flagged), ('passed_low_grade', passed)]:
        take = grp.sample(min(N_EXPLAIN, len(grp)), random_state=42)
        for i, r in enumerate(take.itertuples(), 1):
            exp = explainer.explain_instance(str(r.text)[:2000], proba,
                                             num_features=12,
                                             num_samples=NUM_SAMPLES, labels=(1,))
            for tokn, w in exp.as_list(label=1):
                t = tokn.lower().strip()
                if t and t not in STOP and t.isalpha() and len(t) > 2:
                    rows.append({'group': name, 'token': t, 'weight': w,
                                 'grade': r.holistic_essay_score})
            if i % 10 == 0:
                print('   %s %d/%d' % (name, i, len(take)), flush=True)

    lime_df = pd.DataFrame(rows)
    lime_df.to_csv(os.path.join(RESULTS, 'rq2_lime_tokens.csv'), index=False)

    print()
    top = (lime_df[lime_df.group == 'flagged_high_grade']
           .groupby('token')['weight'].agg(['mean', 'count']).reset_index())
    top = top[top['count'] >= 3].sort_values('mean', ascending=False)
    print('strongest tokens pushing a high-grade authentic essay toward "generated"')
    print(top.head(20).round(4).to_string(index=False))

    mk = set(m for m in MARKERS if ' ' not in m)
    share = top.head(30)['token'].isin(mk).mean()
    print()
    print('share of the top 30 that are pre-registered formal markers: %.0f%%'
          % (100 * share))
    if share >= 0.20:
        print('-> consistent with Part A: the detector reads polish as generation')
    else:
        print('-> the top tokens are NOT mainly formal markers.')
        print('   Part A may still hold, but do not claim LIME corroborates it.')
    return lime_df


def main():
    hu = load_authentic()
    hu = part_a(hu)
    part_b(hu)
    print()
    print('wrote results/rq2_marker_density.csv, rq2_markers_within_grade.csv,')
    print('      rq2_lime_tokens.csv')
    print()
    print('Report this only as a mechanism for RQ2, and only if RQ2 itself shows')
    print('an uneven distribution of false accusations. See results/DECISIONS.md.')


if __name__ == '__main__':
    main()
