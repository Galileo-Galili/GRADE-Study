"""Independent check of the recovered prompt_id.

The recovery used vocabulary exclusive to each prompt's SOURCE PASSAGES.
An independent signal is the ASSIGNMENT FORMAT: prompt 1 asked for a
letter to a state senator, prompt 0 for an explanatory essay. Letters
should contain salutations. If the two signals agree, the labels are
trustworthy.
"""
import sys, io, re, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
csv.field_size_limit(2**31 - 1)

BASE = r'c:\Users\user\Documents\Education Domain'
d = pd.read_csv(BASE + r'\data\llm_detect_prompts.csv')


def squash(t):
    l = re.sub(r'[^a-z]', ' ', str(t).lower())
    return re.sub(r'(.)\1+', r'\1', l)

# Independent signal 1: letter format (prompt 1 only)
LETTER = re.compile(r'\bdear\s+(senator|sir|madam|mr|ms|honorable|state)', re.I)
d['is_letter'] = d['text'].str.contains(LETTER, na=False)

# Independent signal 2: topic-defining nouns absent from the other passage set
CAR = re.compile(r'\b(car|cars|driving|vehicle|vehicles|automobile|traffic|smog)\b', re.I)
ELEC = re.compile(r'\b(electoral college|elector|electors|popular vote)\b', re.I)
d['car_hits'] = d['text'].str.count(CAR)
d['elec_hits'] = d['text'].str.count(ELEC)

known = d[d.prompt_id >= 0]
print('=== agreement with independent signals ===')
print('\n1) letter salutation (expected almost only in prompt 1):')
print(known.groupby('prompt_id')['is_letter'].mean().round(4).to_string())

print('\n2) topic-noun dominance (expected to match prompt_id):')
noun_pid = (known['elec_hits'] > known['car_hits']).astype(int)
agree = (noun_pid == known['prompt_id']).mean()
print(f'   agreement: {100*agree:.1f}%')

dis = known[noun_pid != known['prompt_id']]
print(f'   disagreements: {len(dis)}')
print('\n   sample disagreements:')
for _, r in dis.head(3).iterrows():
    print(f"   [pid={r['prompt_id']} margin={r['margin']} car={r['car_hits']} "
          f"elec={r['elec_hits']}] {str(r['text'])[:110]}")

# Restrict to high-margin and re-check
hi = known[known['margin'] >= 5]
noun_hi = (hi['elec_hits'] > hi['car_hits']).astype(int)
print(f'\n3) restricted to margin>=5 (n={len(hi)}): '
      f'agreement {100*(noun_hi == hi["prompt_id"]).mean():.1f}%')

print('\n=== usable cross-prompt split (margin>=5) ===')
print(hi.groupby('prompt_id')['label'].value_counts().unstack(fill_value=0))
for pid in [0, 1]:
    s = hi[hi.prompt_id == pid]
    n = min(s[s.label == 0].shape[0], s[s.label == 1].shape[0])
    print(f'  prompt {pid}: balanced size would be {2*n} ({n} per class)')
