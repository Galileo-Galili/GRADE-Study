"""Evaluate the five pooled detectors, zero-shot, on the paraphrase slices.

No training happens here. The checkpoints from the pooled condition are loaded
and applied to text they have never seen, which is the deployment condition the
study is about. Length-only and TF-IDF controls are fitted per slice, as
everywhere else in this protocol, so each detector figure can be read against
what a method with no notion of authorship achieves on the same documents.

Appends to results/runs_paraphrase.csv and results/baselines_paraphrase.csv;
both are resumable.
"""

import os
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
MODELS = os.path.join(REPO, 'models')
RESULTS = os.path.join(REPO, 'results')
PREDS = os.path.join(RESULTS, 'predictions')
RUNS = os.path.join(RESULTS, 'runs_paraphrase.csv')
BASE = os.path.join(RESULTS, 'baselines_paraphrase.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN, BATCH, SEED = 256, 8, 42
MODEL_ORDER = ['bert', 'roberta', 'deberta-v3', 'distilbert', 'electra']
SLICES = ['gpt-35', 'claude', 'llama']


def metrics(y, p, pr):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y, p),
        'precision': precision_score(y, p, zero_division=0),
        'recall': recall_score(y, p, zero_division=0),
        'f1': f1_score(y, p, zero_division=0),
        'roc_auc': roc_auc_score(y, pr) if len(set(y)) > 1 else float('nan'),
        'fpr': fp / (fp + tn) if (fp + tn) else float('nan'),
        'fnr': fn / (fn + tp) if (fn + tp) else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


@torch.no_grad()
def predict(model, tok, texts):
    ps, probs = [], []
    for i in range(0, len(texts), BATCH):
        enc = tok(list(texts[i:i + BATCH]), max_length=MAX_LEN, truncation=True,
                  padding='max_length', return_tensors='pt').to(DEVICE)
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ps.extend(torch.argmax(logits, 1).cpu().numpy())
    return np.array(ps), np.array(probs)


def done_keys():
    if not os.path.exists(RUNS):
        return set()
    df = pd.read_csv(RUNS)
    return set(zip(df['model'], df['eval_set']))


def controls(df, name):
    X, y = df['text'].astype(str).tolist(), df['label'].astype(int).tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=SEED, stratify=y)
    out = []

    Ltr = np.array([[len(t)] for t in Xtr])
    Lte = np.array([[len(t)] for t in Xte])
    m = LogisticRegression(max_iter=2000).fit(Ltr, ytr)
    out.append({'corpus': name, 'baseline': 'length_only',
                'accuracy': accuracy_score(yte, m.predict(Lte)),
                'recall': recall_score(yte, m.predict(Lte)),
                'roc_auc': roc_auc_score(yte, m.predict_proba(Lte)[:, 1]),
                'n_test': len(yte)})

    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)
    m = LogisticRegression(max_iter=2000).fit(v.fit_transform(Xtr), ytr)
    Z = v.transform(Xte)
    out.append({'corpus': name, 'baseline': 'tfidf',
                'accuracy': accuracy_score(yte, m.predict(Z)),
                'recall': recall_score(yte, m.predict(Z)),
                'roc_auc': roc_auc_score(yte, m.predict_proba(Z)[:, 1]),
                'n_test': len(yte)})
    return out


def main():
    print('device: %s' % DEVICE)
    sets = {s: pd.read_csv(os.path.join(DATA, 'eval_para_%s.csv' % s))
            for s in SLICES}
    for s, df in sets.items():
        print('  %-8s %6d rows  (%d AI / %d human)'
              % (s, len(df), (df.label == 1).sum(), (df.label == 0).sum()))

    # ---- controls, fitted per slice
    if not os.path.exists(BASE):
        rows = []
        for s, df in sets.items():
            rows += controls(df, s)
        pd.DataFrame(rows).to_csv(BASE, index=False)
        print('\ncontrols written')
    print(pd.read_csv(BASE).round(4).to_string(index=False))

    # ---- detectors, zero-shot
    done = done_keys()
    print('\nalready done: %d' % len(done))
    for name in MODEL_ORDER:
        cd = os.path.join(MODELS, 'pooled_%s_seed42' % name)
        if not os.path.isdir(cd):
            print('[skip] %s: no checkpoint' % name)
            continue
        if all((name, s) in done for s in SLICES):
            print('[skip] %s: all slices done' % name)
            continue

        tok = AutoTokenizer.from_pretrained(cd)
        model = AutoModelForSequenceClassification.from_pretrained(cd).to(DEVICE).eval()
        for s, df in sets.items():
            if (name, s) in done:
                continue
            y = df['label'].astype(int).values
            p, pr = predict(model, tok, df['text'].astype(str).tolist())
            m = metrics(y, p, pr)
            pd.DataFrame({'label': y, 'pred': p, 'prob_ai': pr}).to_csv(
                os.path.join(PREDS, 'preds_para_%s_seed42_%s.csv' % (name, s)),
                index=False)
            row = {'condition': 'pooled', 'model': name, 'eval_set': s,
                   'n_eval': len(df), 'seed': 42, **m}
            pd.DataFrame([row]).to_csv(RUNS, mode='a',
                                       header=not os.path.exists(RUNS), index=False)
            print('  [%s/%s] acc=%.4f recall=%.4f roc=%.4f fpr=%.4f'
                  % (name, s, m['accuracy'], m['recall'], m['roc_auc'], m['fpr']))
        del model, tok
        torch.cuda.empty_cache()

    print('\nDONE')


if __name__ == '__main__':
    main()
