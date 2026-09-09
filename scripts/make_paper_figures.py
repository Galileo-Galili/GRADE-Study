"""Build the two remaining paper figures from measured results.

fig_labels  : the corpus-audit finding (Dataset Collection)
fig_markers : formal-marker density within grade band (Evaluation)

Run from the repository root.  Writes PDF (for LaTeX) and PNG (for preview)
into paper/Assets/.
"""
import glob
import hashlib
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, 'paper', 'Assets')

BLUE = '#1E6CDB'
RED = '#C0392B'
DARK = '#2C3E50'
GREY = '#B8C2CC'
PALE = '#BFD4EE'


def _norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def _hash(s):
    return hashlib.md5(_norm(s).encode()).hexdigest()


def fig_labels():
    """Essays a prompt-vocabulary recovery calls 'car usage', by true assignment."""
    prompts = pd.read_csv(os.path.join(REPO, 'data', 'llm_detect_prompts.csv'))
    prompts['H'] = prompts.text.map(_hash)

    parts = [pd.read_csv(fn, usecols=['text', 'topic'])
             for fn in glob.glob(os.path.join(REPO, 'data', 'loto', 'fold_00_*.csv'))]
    truth = pd.concat(parts)
    truth['H'] = truth.text.map(_hash)
    truth = truth.drop_duplicates('H')

    joined = prompts.merge(truth[['H', 'topic']], on='H', how='inner')
    recovered = joined[joined.prompt_id == 0]
    share = recovered.topic.value_counts() / len(recovered) * 100

    fig, ax = plt.subplots(figsize=(3.35, 2.75), dpi=400)
    labels = [t.replace('"', '') for t in share.index]
    vals = share.values
    colours = [BLUE if t == 'Car-free cities' else GREY for t in share.index]
    y = np.arange(len(labels))[::-1]

    ax.barh(y, vals, color=colours, height=0.72, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.6, yy, '%.1f%%' % v, va='center', fontsize=5.4, color=DARK,
                fontweight='bold' if v == vals.max() else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.set_xlabel('Share of the essays recovery calls "car usage" (%)', fontsize=6.2)
    ax.set_xlim(0, vals.max() * 1.22)
    ax.tick_params(axis='x', labelsize=5.6)
    ax.tick_params(axis='y', length=0)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)
    ax.set_title('Only 23% of the essays a prompt recovery calls\n'
                 '"car usage" actually answer that assignment',
                 fontsize=6.5, fontweight='bold', pad=5, linespacing=1.3)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(ASSETS, 'fig_labels.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(ASSETS, 'fig_labels.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print('fig_labels : intended share %.1f%%, spans %d assignments'
          % (vals[0], len(share)))


def fig_markers():
    """Formal-marker density, flagged versus passed, within each grade band."""
    m = pd.read_csv(os.path.join(REPO, 'results', 'rq2_markers_within_grade.csv'))

    fig, ax = plt.subplots(figsize=(3.35, 2.55), dpi=400)
    x = np.arange(len(m))
    w = 0.36

    ax.bar(x - w / 2, m.markers_flagged, w, label='flagged as AI', color=RED, zorder=3)
    ax.bar(x + w / 2, m.markers_passed, w, label='passed', color=PALE, zorder=3)

    for i, row in m.iterrows():
        top = max(row.markers_flagged, row.markers_passed)
        ax.text(i, top + 0.16, '+%.2f' % row.difference, ha='center',
                fontsize=5.3, color=DARK, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(m.grade.astype(int), fontsize=6.0)
    ax.set_xlabel('Holistic grade band', fontsize=6.3)
    ax.set_ylabel('Formal markers per 1,000 words', fontsize=6.3)
    ax.set_ylim(0, m.markers_flagged.max() * 1.25)
    ax.tick_params(axis='y', labelsize=5.8)
    ax.tick_params(axis='x', length=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', color='#EFEFEF', lw=0.5, zorder=0)
    ax.legend(fontsize=5.6, frameon=False, loc='upper left', handlelength=1.1,
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_title('Within every grade band, the essays flagged\n'
                 'as AI are the more formally written ones',
                 fontsize=6.5, fontweight='bold', pad=5, linespacing=1.3)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(ASSETS, 'fig_markers.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(ASSETS, 'fig_markers.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print('fig_markers: %d bands, every difference positive: %s'
          % (len(m), bool((m.difference > 0).all())))


def fig_margins():
    """Per-fold detector accuracy against the word-frequency control."""
    runs = pd.read_csv(os.path.join(REPO, 'results', 'loto_runs.csv'))
    base = pd.read_csv(os.path.join(REPO, 'results', 'loto_baselines.csv'))

    det = runs[(runs.model == 'deberta-v3') & (runs.seed == 42)]
    det = det[['fold', 'held_out_topic', 'accuracy']]
    tfidf = base[base.baseline == 'tfidf'][['fold', 'accuracy']]
    tfidf = tfidf.rename(columns={'accuracy': 'control'})

    m = det.merge(tfidf, on='fold')
    m['margin'] = m.accuracy - m.control
    m['control_wins'] = m.margin <= 0
    m = m.sort_values('margin').reset_index(drop=True)

    short = {
        'Does the electoral college work?': 'Electoral college',
        'Mandatory extracurricular activities': 'Mandatory extracurriculars',
        '"A Cowboy Who Rode the Waves"': 'A Cowboy Who Rode the Waves',
        'Grades for extracurricular activities': 'Grades for extracurriculars',
        'Facial action coding system': 'Facial action coding',
    }
    m['label'] = m.held_out_topic.map(lambda t: short.get(t, t))

    fig, ax = plt.subplots(figsize=(3.35, 2.9), dpi=400)
    n_lost = int(m.control_wins.sum())
    ax.axhspan(-0.5, n_lost - 0.5, color='#FDECEC', zorder=0)

    for i, row in m.iterrows():
        ax.plot([row.control * 100, row.accuracy * 100], [i, i],
                color='#DDDDDD', lw=1.3, zorder=1, solid_capstyle='round')
        ax.scatter(row.control * 100, i, marker='D', s=22, zorder=3,
                   color=RED if row.control_wins else '#B0B0B0',
                   edgecolor='white', linewidth=0.4,
                   label='word-frequency control' if i == 0 else None)
        ax.scatter(row.accuracy * 100, i, s=22, zorder=3, color=BLUE,
                   edgecolor='white', linewidth=0.4,
                   label='DeBERTa-v3-small' if i == 0 else None)

    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.label, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy on the held-out assignment (%)', fontsize=6.2)
    ax.set_xlim(92, 101)
    ax.tick_params(axis='x', labelsize=5.6)
    ax.tick_params(axis='y', length=0)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='x', color='#EFEFEF', lw=0.5, zorder=0)
    ax.legend(fontsize=5.5, frameon=False, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.30), handletextpad=0.3,
              columnspacing=1.2, borderaxespad=0.0)
    ax.set_title('A control that cannot recognise authorship matches\n'
                 'or beats the detector on 9 of the 15 assignments',
                 fontsize=6.5, fontweight='bold', pad=5, linespacing=1.3)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(ASSETS, 'fig_margins.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(ASSETS, 'fig_margins.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print('fig_margins: control matches or beats the detector on %d of %d folds'
          % (n_lost, len(m)))


def fig_calibration():
    """Accuracy against ROC-AUC per fold, both encoders."""
    runs = pd.read_csv(os.path.join(REPO, 'results', 'loto_runs.csv'))
    deb = runs[(runs.model == 'deberta-v3') & (runs.seed == 42)].sort_values('fold')
    rob = runs[(runs.model == 'roberta') & (runs.seed == 42)].sort_values('fold')

    fig, ax = plt.subplots(figsize=(3.35, 2.55), dpi=400)
    x = np.arange(1, 16)

    ax.plot(x, deb.roc_auc * 100, '-', color=DARK, lw=1.4, zorder=4)
    ax.plot(x, rob.roc_auc * 100, '-', color='#7E8C9A', lw=1.4, zorder=4)
    ax.plot(x, deb.accuracy * 100, 'o-', color=BLUE, lw=0.9, ms=2.6, zorder=3)
    ax.plot(x, rob.accuracy * 100, 'o-', color='#7FAAE8', lw=0.9, ms=2.6, zorder=3)

    ax.text(8.0, 100.6, 'ROC-AUC  (both encoders)', fontsize=5.7, color=DARK,
            ha='center', fontweight='bold')
    ax.text(6.4, 95.3, 'accuracy', fontsize=5.7, color=BLUE, ha='center',
            fontweight='bold')
    ax.annotate('threshold fails here', xy=(3.0, 90.4), xytext=(4.5, 91.7),
                fontsize=5.4, color=RED, ha='left',
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=0.7))

    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(89.0, 101.7)
    ax.set_xticks([1, 5, 10, 15])
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel('Held-out assignment (fold)', fontsize=6.2)
    ax.set_ylabel('%', fontsize=6.2)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', color='#EFEFEF', lw=0.5, zorder=0)
    ax.set_title('Ranking survives the shift; the threshold does not',
                 fontsize=6.5, fontweight='bold', pad=5)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(ASSETS, 'fig_calibration.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(ASSETS, 'fig_calibration.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print('fig_calibration: min AUC deb=%.2f rob=%.2f, min acc deb=%.2f rob=%.2f'
          % (deb.roc_auc.min() * 100, rob.roc_auc.min() * 100,
             deb.accuracy.min() * 100, rob.accuracy.min() * 100))


if __name__ == '__main__':
    fig_labels()
    fig_markers()
    fig_margins()
    fig_calibration()
