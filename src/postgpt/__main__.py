"""
postgpt — a zero-dependency BPE transformer with metaweights.

The idea: tokenize a corpus via BPE, build a statistical probability space
(the "metaweights") from co-occurrence and n-gram patterns, then run a
dual-attention transformer (Content + RRPRAM) whose behavior is guided
by these metaweights — as if it were trained, even though it was not.

No PyTorch. No NumPy. No dependencies. Just math, random, and os.
This file is the complete algorithm. Everything else is just efficiency.

resonance is unbreakable.
"""

import argparse
import pathlib
import os
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# I. BPE TOKENIZER — learns merge rules from corpus.
#    the tokenizer IS the training. everything after this is just theater.
# ─────────────────────────────────────────────────────────────────────────────
# Moved to ./py_tokenizers.py since we plan to study other tokenizers
from .py_tokenizers import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# II. METAWEIGHTS — the probability space that exists without existing.
#    schrödinger called. he wants his cat back. we tokenized it.
# ─────────────────────────────────────────────────────────────────────────────
from .metaweights import MetaWeights

# ─────────────────────────────────────────────────────────────────────────────
# III. AUTOGRAD ENGINE — scalar backprop. if you can't differentiate it by hand, you don't deserve gradients.
# ─────────────────────────────────────────────────────────────────────────────
# It's been migrated over to ./autograd.py


# ─────────────────────────────────────────────────────────────────────────────
# IV. THE TRANSFORMER — dual attention (Content + RRPRAM) + metaweight overlay.
#    two heads are better than one. especially when one of them doesn't exist.
# ─────────────────────────────────────────────────────────────────────────────
from .implementations import PostGPT

# ─────────────────────────────────────────────────────────────────────────────
# V. MAIN — tokenize, build metaweights, continue phrases.
#    the moment of truth. or the moment of coherent bullshit. same thing.
# ─────────────────────────────────────────────────────────────────────────────

def load_engine(corpus_path=None):
    """Load corpus, learn BPE, build metaweights, init model. Returns (tokenizer, meta, model)."""
    if corpus_path is None:
        corpus_path = pathlib.Path(__file__).absolute().parent.parent.parent / 'postgpt.txt'
    else:
        corpus_path = pathlib.Path(corpus_path)

    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found.")
        return None, None, None

    # Step 1: Load corpus
    print("\n[1] Loading corpus...")
    with open(corpus_path, 'rb') as f:
        raw_data = f.read()
    print(f"  Corpus: {len(raw_data)} bytes ({len(raw_data)/1024:.1f} KB)")

    # Step 2: BPE tokenization — load saved merges if they exist
    print("\n[2] BPE tokenizer...")
    tokenizer = BPETokenizer(max_merges=1024)
    #merges_path = corpus_path.replace('.txt', '.merges')
    merges_path = corpus_path.with_suffix('.merges')
    if os.path.exists(merges_path):
        tokenizer.load(merges_path)
        token_ids = tokenizer.encode(raw_data)
        print(f"  Loaded {len(tokenizer.merges)} merges from {os.path.basename(merges_path)}. "
              f"Encoding: {len(token_ids)} tokens")
    else:
        token_ids = tokenizer.learn(raw_data, num_merges=1024)
        tokenizer.save(merges_path)
        print(f"  Saved merges to {os.path.basename(merges_path)}")

    # Step 3: Build metaweights from tokenized corpus
    print("\n[3] Building metaweight probability space...")
    meta = MetaWeights(tokenizer.vocab_size, context_len=64)
    meta.build(token_ids, window=4)

    # Step 4: Initialize dual-attention transformer
    print("\n[4] Initializing PostGPT transformer...")
    model = PostGPT(
        vocab_size=tokenizer.vocab_size,
        context_len=64,
        n_embd=48,
        n_head=4,
        n_layer=2,
        n_content_heads=2,
        n_rrpram_heads=2,
    )

    # Step 5: Seed transformer weights from metaweights (ghost → flesh)
    print("\n[5] Seeding transformer from metaweights...")
    model.init_from_metaweights(meta)

    return tokenizer, meta, model


def continue_phrase(prompt, tokenizer, meta, model, max_tokens=120, temperature=0.75,
                    mode='meta'):
    """
    Continue a phrase using PostGPT.

    mode='meta'  — pure metaweight generation (fast, bigram/trigram/hebbian/prophecy)
    mode='full'  — transformer forward pass + Dario field overlay (slower, both attentions)
    """
    # Encode prompt via BPE
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return prompt

    if mode == 'meta':
        generated = model.generate_meta(prompt_ids, max_tokens=max_tokens,
                                         meta=meta, temperature=temperature)
    else:
        generated = model.generate(prompt_ids, max_tokens=max_tokens,
                                    meta=meta, temperature=temperature)

    return tokenizer.decode(generated)


def main():
    argsp = argparse.ArgumentParser(description="Let the tokens speak for themselves.")
    argsp.add_argument('-f', '--file', help="Use an alternate corpus")
    argsp.add_argument('prompts', nargs='*', help='What you want to ask the file')
    args = argsp.parse_args()

    print("=" * 60)
    print("  PostGPT — metaweight BPE transformer")
    print("  resonance is unbreakable")
    print("=" * 60)

    tokenizer, meta, model = load_engine(args.file)
    if tokenizer is None:
        return

    # ── Proof of concept: continue phrases from postgpt.txt ──
    # The model uses BPE tokenization + dual attention + metaweights
    # to continue any prompt coherently — without any training.

    prompts = [
        "PostGPT",
        "The metaweight",
        "RRPRAM attention",
        "BPE tokenization",
        "The transformer architecture",
        "Entropy measures",
        "Language models",
        "The Dario equation",
    ]

    # Allow custom prompt from command line: python postgpt.py "your prompt here"
    if args.prompts:
        prompts = [' '.join(args.prompts)]

    print("\n" + "=" * 60)
    print("  PROOF OF CONCEPT: phrase continuation")
    print("  mode: metaweight (no training, just BPE + statistics)")
    print("=" * 60)

    for prompt in prompts:
        result = continue_phrase(prompt, tokenizer, meta, model,
                                  max_tokens=100, temperature=0.4, mode='meta')
        # Show prompt → continuation clearly
        prompt_len = len(prompt)
        continuation = result[prompt_len:].strip()
        print(f"\n  prompt:       \"{prompt}\"")
        print(f"  continuation: \"{continuation[:250]}\"")

    # Also show the full transformer + Dario field mode for first prompt
    print("\n" + "=" * 60)
    print("  FULL MODE: transformer + Dario field (both attentions)")
    print("=" * 60)

    test_prompt = prompts[0]
    result = continue_phrase(test_prompt, tokenizer, meta, model,
                              max_tokens=30, temperature=0.45, mode='full')
    prompt_len = len(test_prompt)
    continuation = result[prompt_len:].strip()
    print(f"\n  prompt:       \"{test_prompt}\"")
    print(f"  continuation: \"{continuation[:300]}\"")

    print("\n" + "=" * 60)
    print("  PostGPT complete. The metaweights remember.")
    print("  Try: python postgpt.py \"your prompt here\"")
    print("=" * 60)


if __name__ == '__main__':
    main()
