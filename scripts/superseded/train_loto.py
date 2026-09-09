"""Leave-one-assignment-out sweep.

Run from the repository root with the tf1 environment:

    python scripts/train_loto.py

Resumable. Each finished run appends one row to results/loto_runs.csv and is
skipped on restart, so the sweep can be stopped and continued freely.

Two stages, per Section "The GRADE protocol":

  Stage 1   all 15 folds at seed 42.
  Stage 2   the best and worst folds from stage 1, repeated at seeds 123
            and 2024, so a spread can be told apart from run-to-run
            variation. Run stage 1 first, then set STAGE = 2.

Predictions are written with the essay uid alongside the verdict, so the
grade attribution in analyse_loto.py joins on identity rather than on row
order.
"""

import gc
import os
import random
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ----------------------------------------------------------------- config
STAGE = 1                    # 1 = all folds at seed 42; 2 = extremes at more seeds
MODEL = 'deberta-v3'         # strongest cross-assignment encoder in pilot work
EXTRA_SEEDS = [123, 2024]    # used by stage 2
SAVE_CHECKPOINTS = False     # 15 checkpoints is ~6 GB; predictions suffice

# One checkpoint is kept so the RQ2 mechanism check can be run afterwards:
# scripts/explain_rq2.py needs a live model to perturb essays against. Fold 0
# has the largest held-out set, so it yields the most false positives to
# explain. About 570 MB.
KEEP_CHECKPOINT_FOLDS = [0]

# ---------------------------------------------------------------------------
# Side probe: the Abstract Paraphrasing corpus (304 PhD abstracts, 152:152).
#
# Qorich and El Ouazzani report 97.38% on this exact corpus under five-fold
# cross-validation, which trains on four fifths of it before testing. Scoring
# it here costs a few seconds per fold and measures the same corpus without
# training on it at all.
#
# THIS IS NOT ONE OF THE PAPER'S TWO RESEARCH QUESTIONS. It changes both the
# register and the generating task relative to training, so it is not a
# controlled comparison. It goes in the paper only as a single corroborating
# sentence in the related-work section, and only if the pre-registered rule in
# results/DECISIONS.md is met. Otherwise it is left out entirely.
# ---------------------------------------------------------------------------
EVAL_ABSTRACTS = True

CONFIG = {
    'max_len': 256,
    'lr': 2e-5,
    'batch_size': 8,
    'epochs': 3,             # validation peaked at epoch 2 in pilot runs
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOTO = os.path.join(REPO, 'data', 'loto')
RESULTS = os.path.join(REPO, 'results')
PREDS = os.path.join(RESULTS, 'loto_predictions')
MODELS = os.path.join(REPO, 'models', 'loto')
RUNS = os.path.join(RESULTS, 'loto_runs.csv')
ABSTRACTS = os.path.join(REPO, 'data', 'eval_abstract.csv')
ABS_RUNS = os.path.join(RESULTS, 'loto_abstract_probe.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ZOO = {
    'bert':       'bert-base-uncased',
    'roberta':    'roberta-base',
    'deberta-v3': os.path.join(REPO, 'models', '_base', 'deberta-v3-small'),
    'distilbert': 'distilbert-base-uncased',
    'electra':    os.path.join(REPO, 'models', '_base', 'electra-base-discriminator'),
}
CANONICAL = {
    'bert': 'bert-base-uncased', 'roberta': 'roberta-base',
    'deberta-v3': 'microsoft/deberta-v3-small',
    'distilbert': 'distilbert-base-uncased',
    'electra': 'google/electra-base-discriminator',
}


# ----------------------------------------------------------------- helpers
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class TextDS(Dataset):
    def __init__(self, df):
        self.t = df['text'].astype(str).tolist()
        self.y = df['label'].astype(int).tolist()

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.y[i]


def loader(df, bs, shuffle):
    return DataLoader(TextDS(df), batch_size=bs, shuffle=shuffle)


def encode(tok, texts, max_len):
    return tok(list(texts), max_length=max_len, truncation=True,
               padding='max_length', return_tensors='pt').to(DEVICE)


@torch.no_grad()
def predict(model, tok, df, cfg, desc='eval'):
    model.eval()
    ys, ps, probs = [], [], []
    for texts, y in tqdm(loader(df, cfg['batch_size'], False), desc=desc, leave=False):
        logits = model(**encode(tok, texts, cfg['max_len'])).logits
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ps.extend(torch.argmax(logits, 1).cpu().numpy())
        ys.extend(np.asarray(y))
    return np.array(ys), np.array(ps), np.array(probs)


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


def train_one(hf_id, seed, cfg, train_df, val_df, tag):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    tl = loader(train_df, cfg['batch_size'], True)

    best_acc, best_state = -1.0, None
    for ep in range(cfg['epochs']):
        model.train()
        t0, tot = time.time(), 0.0
        for texts, y in tqdm(tl, desc='%s ep%d/%d' % (tag, ep + 1, cfg['epochs']),
                             leave=False):
            yt = torch.as_tensor(y).to(DEVICE)
            opt.zero_grad()
            out = model(**encode(tok, texts, cfg['max_len']), labels=yt)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        yv, pv, _ = predict(model, tok, val_df, cfg, desc='val')
        vacc = float((yv == pv).mean())
        print('   ep%d loss=%.4f val_acc=%.4f (%.0fs)'
              % (ep + 1, tot / max(len(tl), 1), vacc, time.time() - t0), flush=True)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model, tok, best_acc


def done_keys():
    if not os.path.exists(RUNS):
        return set()
    d = pd.read_csv(RUNS)
    return set(zip(d['model'], d['fold'], d['seed']))


def pick_extremes():
    """Best and worst fold from stage 1, by accuracy at seed 42."""
    d = pd.read_csv(RUNS)
    s1 = d[(d['seed'] == 42) & (d['model'] == MODEL)]
    if len(s1) < 15:
        raise SystemExit('stage 2 needs all 15 stage-1 folds; have %d' % len(s1))
    return [int(s1.loc[s1['accuracy'].idxmin(), 'fold']),
            int(s1.loc[s1['accuracy'].idxmax(), 'fold'])]


# ----------------------------------------------------------------- sweep
def main():
    for d in (RESULTS, PREDS, MODELS):
        os.makedirs(d, exist_ok=True)

    print('device      : %s' % DEVICE)
    if DEVICE != 'cuda':
        print('*** NO GPU DETECTED - this will be impractically slow. ***')
    print('detector    : %s' % MODEL)
    print('config      : %s' % CONFIG)

    folds = pd.read_csv(os.path.join(LOTO, 'folds.csv'))

    if STAGE == 1:
        jobs = [(int(r.fold), 42) for r in folds.itertuples()]
    else:
        jobs = [(f, s) for f in pick_extremes() for s in EXTRA_SEEDS]
    print('stage %d     : %d run(s) planned' % (STAGE, len(jobs)))
    print()

    hf_id = ZOO[MODEL]
    for fold, seed in jobs:
        if (MODEL, fold, seed) in done_keys():
            print('[skip] fold %02d seed %d already done' % (fold, seed))
            continue

        topic = folds.loc[folds.fold == fold, 'held_out_topic'].iloc[0]
        tr = pd.read_csv(os.path.join(LOTO, 'fold_%02d_train.csv' % fold))
        va = pd.read_csv(os.path.join(LOTO, 'fold_%02d_val.csv' % fold))
        te = pd.read_csv(os.path.join(LOTO, 'fold_%02d_test.csv' % fold))

        tag = 'f%02d s%d' % (fold, seed)
        print('=== fold %02d  seed %d  held out: %s ===' % (fold, seed, topic))
        print('    train=%d val=%d test=%d' % (len(tr), len(va), len(te)),
              flush=True)
        try:
            model, tok, best_val = train_one(hf_id, seed, CONFIG, tr, va, tag)
        except Exception as ex:
            print('[FAIL] %s: %s: %s' % (tag, type(ex).__name__, ex))
            continue

        y, p, pr = predict(model, tok, te, CONFIG, desc='test')
        m = metrics(y, p, pr)

        pd.DataFrame({'uid': te['uid'].values, 'label': y,
                      'pred': p, 'prob_ai': pr}).to_csv(
            os.path.join(PREDS, 'preds_f%02d_%s_seed%d.csv' % (fold, MODEL, seed)),
            index=False)

        pd.DataFrame([{'model': MODEL, 'hf_id': CANONICAL[MODEL], 'fold': fold,
                       'held_out_topic': topic, 'seed': seed,
                       'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                       'best_val_accuracy': best_val,
                       'epochs': CONFIG['epochs'], 'lr': CONFIG['lr'],
                       'batch_size': CONFIG['batch_size'],
                       'max_len': CONFIG['max_len'], **m}]).to_csv(
            RUNS, mode='a', header=not os.path.exists(RUNS), index=False)

        print('    acc=%.4f recall=%.4f roc=%.4f fpr=%.4f fnr=%.4f'
              % (m['accuracy'], m['recall'], m['roc_auc'], m['fpr'], m['fnr']),
              flush=True)

        # ---- side probe on the abstracts, while the model is still loaded
        if EVAL_ABSTRACTS and os.path.exists(ABSTRACTS):
            ab = pd.read_csv(ABSTRACTS)
            ya, pa, pra = predict(model, tok, ab, CONFIG, desc='abstracts')
            ma = metrics(ya, pa, pra)
            pd.DataFrame([{'model': MODEL, 'fold': fold, 'seed': seed,
                           'n_eval': len(ab), **ma}]).to_csv(
                ABS_RUNS, mode='a', header=not os.path.exists(ABS_RUNS),
                index=False)
            print('    [abstracts] acc=%.4f recall=%.4f roc=%.4f fpr=%.4f'
                  % (ma['accuracy'], ma['recall'], ma['roc_auc'], ma['fpr']),
                  flush=True)

        if SAVE_CHECKPOINTS or (fold in KEEP_CHECKPOINT_FOLDS and seed == 42):
            cd = os.path.join(MODELS, 'f%02d_%s_seed%d' % (fold, MODEL, seed))
            os.makedirs(cd, exist_ok=True)
            model.save_pretrained(cd, safe_serialization=True)
            tok.save_pretrained(cd)
            print('    checkpoint kept -> %s' % cd, flush=True)

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print('STAGE %d COMPLETE' % STAGE)
    if os.path.exists(RUNS):
        d = pd.read_csv(RUNS)
        print(d[['fold', 'held_out_topic', 'seed', 'accuracy', 'recall',
                 'roc_auc', 'fpr']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
