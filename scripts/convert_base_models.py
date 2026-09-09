"""Convert the two base checkpoints that ship without safetensors.

microsoft/deberta-v3-small and google/electra-base-discriminator publish only
pytorch_model.bin. transformers >= 4.56 refuses to torch.load a .bin unless
torch >= 2.6 (CVE-2025-32434), so on torch 2.5.x both models fail to load with:

    ValueError: Due to a serious vulnerability issue in `torch.load` ...

Rather than upgrade torch underneath a working CUDA environment, we convert the
weights to safetensors once, into models/_base/, and load from there. The
weights are unchanged: only the container format differs.

Run from the repository root:

    python scripts/convert_base_models.py

Verified after conversion: the resulting tokenizer is token- and id-identical to
the reference sentencepiece tokenizer, and the model completes a forward and
backward pass.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'models', '_base')

TARGETS = [
    ('microsoft/deberta-v3-small', 'deberta-v3-small'),
    ('google/electra-base-discriminator', 'electra-base-discriminator'),
]


def already_converted(path):
    return (os.path.isdir(path)
            and any(f.endswith('.safetensors') for f in os.listdir(path)))


def convert(repo_id, slug):
    dst = os.path.join(OUT, slug)
    if already_converted(dst):
        print('[skip] %s already converted' % slug)
        return dst

    print('=== converting %s' % repo_id)
    bin_path = hf_hub_download(repo_id, 'pytorch_model.bin')
    state = torch.load(bin_path, map_location='cpu', weights_only=True)
    # save_file requires contiguous tensors and rejects shared storage
    state = {k: v.contiguous().clone()
             for k, v in state.items() if isinstance(v, torch.Tensor)}

    os.makedirs(dst, exist_ok=True)
    save_file(state, os.path.join(dst, 'model.safetensors'),
              metadata={'format': 'pt'})
    AutoConfig.from_pretrained(repo_id).save_pretrained(dst)
    AutoTokenizer.from_pretrained(repo_id).save_pretrained(dst)
    print('  saved -> %s' % dst)
    return dst


def verify(dst, repo_id):
    """Model trains, and tokenization matches the reference implementation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(
        dst, num_labels=2).to(device)
    tok = AutoTokenizer.from_pretrained(dst)

    enc = tok(['a test essay about cars.', 'another one.'], max_length=256,
              truncation=True, padding='max_length', return_tensors='pt'
              ).to(device)
    out = model(**enc, labels=torch.tensor([0, 1]).to(device))
    out.loss.backward()

    ref = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    probe = ("The Electoral College should be abolished; it's undemocratic. "
             "Cars, however, reduce driving stress.")
    same = tok(probe)['input_ids'] == ref(probe)['input_ids']

    print('  verify: loss=%.4f  tokenizer_matches_reference=%s'
          % (out.loss.item(), same))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return same


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for repo_id, slug in TARGETS:
        dst = convert(repo_id, slug)
        if not verify(dst, repo_id):
            print('  ERROR: tokenization differs from reference for %s' % slug)
            ok = False
    print('CONVERT DONE' if ok else 'CONVERT FINISHED WITH ERRORS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
