"""Count body words in the .tex, excluding refs/appendix/comments/macros
so we can track against the CFP's 2500-4500 window."""
import re, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# default: the paper in this repo, resolved from this file's location so the
# script works from any working directory and on any machine
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT = os.path.join(_REPO, 'paper', 'iciser2026_short.tex')

p = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
t = open(p, encoding='utf-8').read()

# cut bibliography onward (references excluded from count)
t = re.split(r'\\begin\{thebibliography\}', t)[0]
# drop preamble before \begin{document}
t = t.split(r'\begin{document}', 1)[-1]
# strip comment lines
t = '\n'.join(l for l in t.split('\n') if not l.lstrip().startswith('%'))

def seg(name):
    """Extract abstract separately."""
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
    return m.group(1) if m else ''

abstract = seg('abstract')
body = t.replace(abstract, '')

def words(s):
    s = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', s, flags=re.S)
    s = re.sub(r'\\[a-zA-Z@]+\*?(\[[^\]]*\])?', ' ', s)   # macros
    s = re.sub(r'[{}$&\\_^~#]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return [w for w in s.split() if re.search(r'[A-Za-z]', w)]

aw, bw = words(abstract), words(body)
print(f'abstract words : {len(aw):>5}   (CFP max 150)')
print(f'body words     : {len(bw):>5}   (CFP 2500-4500)')
status = 'OK' if 2500 <= len(bw) <= 4500 else ('UNDER' if len(bw) < 2500 else 'OVER')
print(f'status         : {status}')
if len(bw) < 2500:
    print(f'                 need +{2500-len(bw)} words')
