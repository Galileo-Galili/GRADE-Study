"""Recover prompt_id by matching against terms EXCLUSIVE to each prompt's
source passages. Students were told to cite the passages, so distinctive
proper nouns and figures should betray which passage set they read."""
import sys, io, re, csv, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
csv.field_size_limit(2**31 - 1)

BASE = r'c:\Users\user\Documents\Education Domain'
prompts = pd.read_csv(BASE + r'\datasets\llm_detect\train_prompts.csv')
df = pd.read_csv(BASE + r'\data\llm_detect_clean.csv')

def words(t):
    return set(re.findall(r'[a-z]{4,}', str(t).lower()))

w0 = words(prompts.iloc[0]['source_text'] + ' ' + prompts.iloc[0]['instructions'])
w1 = words(prompts.iloc[1]['source_text'] + ' ' + prompts.iloc[1]['instructions'])

only0 = w0 - w1
only1 = w1 - w0
print(f'vocab: prompt0={len(w0)} prompt1={len(w1)}')
print(f'exclusive: prompt0={len(only0)} prompt1={len(only1)}')

# Corruption-robust normalisation: collapse doubled chars so 'senaaor'->'senaor'
def squash(t):
    l = re.sub(r'[^a-z]', ' ', str(t).lower())
    return re.sub(r'(.)\1+', r'\1', l)

sq0 = {re.sub(r'(.)\1+', r'\1', w) for w in only0}
sq1 = {re.sub(r'(.)\1+', r'\1', w) for w in only1}
sq0, sq1 = sq0 - sq1, sq1 - sq0

print('\nsample exclusive terms')
print(' prompt0:', sorted(list(only0))[:25])
print(' prompt1:', sorted(list(only1))[:25])

scores = []
for t in df['text']:
    toks = set(squash(t).split())
    scores.append((len(toks & sq0), len(toks & sq1)))

c = pd.Series([s[0] for s in scores])
e = pd.Series([s[1] for s in scores])
pid = pd.Series(0, index=df.index)
pid[e > c] = 1
pid[c == e] = -1

vc = pid.value_counts().to_dict()
print('\nassignment:', vc)
print('unassigned:', vc.get(-1, 0), f'({100*vc.get(-1,0)/len(df):.1f}%)')

# margin: how decisive is each assignment?
margin = (c - e).abs()
print(f'\nmargin: median={margin.median():.0f} '
      f'p10={margin.quantile(.10):.0f} p25={margin.quantile(.25):.0f}')
print('assignments with margin >= 3:', int((margin >= 3).sum()),
      f'({100*(margin>=3).sum()/len(df):.1f}%)')

df['prompt_id'] = pid
df['margin'] = margin
conf = df[df['margin'] >= 3]
print('\nconfident subset:', len(conf))
print(conf.groupby('prompt_id')['label'].value_counts().unstack(fill_value=0))

df.to_csv(BASE + r'\data\llm_detect_prompts.csv', index=False)
print('\nwrote data/llm_detect_prompts.csv')
