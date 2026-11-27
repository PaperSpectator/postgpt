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
import json

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
from . import DEFAULT_ENGINE_CONFIG

def load_engine(corpus_path=None, tokenizer_path=None, engine_config=None, context_len=64, metaweights_window=4):
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
    if tokenizer_path:
        tokenizer_path = pathlib.Path(tokenizer_path)
        match tokenizer_path.suffix:
            case '.merges':
                print("Loading tokenizer from .merges bin")
                if not os.path.exists(tokenizer_path):
                    raise ValueError("Tokenizer must exist!")
                tokenizer = BPETokenizer(max_merges=1024)
                tokenizer.load(tokenizer_path)
                vocab_size = tokenizer.vocab_size
                token_ids = tokenizer.encode(raw_data)
                print(f"  Loaded {len(tokenizer.merges)} merges from {os.path.basename(tokenizer_path)}. "
                f"Encoding: {len(token_ids)} tokens")
            case '.json':
                print("Loading tokenizer from *.json")
                from tokenizers import Tokenizer
                if not os.path.exists(tokenizer_path):
                    raise ValueError("Tokenizer must exist!")
                tokenizer  = Tokenizer.from_file(str(tokenizer_path))
                vocab_size = tokenizer.get_vocab_size()
                token_ids  = tokenizer.encode(raw_data.decode('utf-8')).ids
                #print(token_ids.ids)
            case _:
                print("Loading tokenizer from HuggingFace")
                from tokenizers import Tokenizer
                tokenizer  = Tokenizer.from_pretrained(tokenizer_path)
                vocab_size = tokenizer.get_vocab_size()
                token_ids  = tokenizer.encode(raw_data.decode('utf-8')).ids

    else:
        tokenizer = BPETokenizer(max_merges=1024)
        #merges_path = corpus_path.replace('.txt', '.merges')
        merges_path = corpus_path.with_suffix('.merges')
        if os.path.exists(merges_path):
            tokenizer.load(merges_path)
            vocab_size = tokenizer.vocab_size
            # TODO: skip encoding the corpus just to count how many IDs we have since we aren't learning from it
            token_ids = tokenizer.encode(raw_data)
            print(f"  Loaded {len(tokenizer.merges)} merges from {os.path.basename(merges_path)}. "
                f"Encoding: {len(token_ids)} tokens")
        else:
            token_ids = tokenizer.learn(raw_data, num_merges=1024)
            vocab_size = tokenizer.vocab_size
            tokenizer.save(merges_path)
            print(f"  Saved merges to {os.path.basename(merges_path)}")

    # Step 3: Build metaweights from tokenized corpus
    print("\n[3] Building metaweight probability space...")
    meta = MetaWeights(vocab_size, context_len=context_len)
    meta.build(token_ids, window=metaweights_window)
    mw_json = corpus_path.with_suffix('.mw.json')
    with open(mw_json, 'w', encoding='utf-8') as mw_f:
        json.dump(meta.todict(), mw_f, indent=4)

    # Step 4: Initialize dual-attention transformer
    print("\n[4] Initializing PostGPT transformer...")
    if not engine_config:
        engine_config = DEFAULT_ENGINE_CONFIG.copy()
        engine_config['context_len'] = context_len
    model = PostGPT(
        vocab_size=vocab_size,
        **engine_config
    )

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
    try:
        iterator = iter(prompt_ids)
    except TypeError:
        # we got a huggingface tokenizer
        print("Prompt ids encoding not initially iterable because huggingface tokenizer")
        print("Grabbing the ids field")
        prompt_ids = prompt_ids.ids

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
    argsp.add_argument('-t', '--tokenizer', help="Talk with non-dynamic tokenizer", default=None)
    argsp.add_argument('-e', '--engine', help="Load engine parameters from a JSON file", default=None)
    argsp.add_argument('-c', '--context-len', help="Set the context size", default=64, type=int)
    argsp.add_argument('-w', '--metaweights-window', help="Set the metaweights window", default=4, type=int)
    argsp.add_argument('-m', '--continuation-mode', default='both', help="The type of continuation to use", choices=('both', 'meta', 'full'))
    argsp.add_argument('--seed', default=42, help="Seed for generation")
    argsp.add_argument('prompts', nargs='*', help='What you want to ask the file')
    args = argsp.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  PostGPT — metaweight BPE transformer")
    print("  resonance is unbreakable")
    print("=" * 60)
    engine_config = None
    if args.engine and os.path.exists(args.engine):
        with open(args.engine) as f:
            engine_config = json.load(f)

    loader_args = {
        'tokenizer_path': args.tokenizer,
        'engine_config': engine_config,
        'context_len': args.context_len,
        'metaweights_window': args.metaweights_window
    }
    tokenizer, meta, model = load_engine(args.file, **loader_args)
    if tokenizer is None:
        return

    # Step 5: Seed transformer weights from metaweights (ghost → flesh)
    if args.continuation_mode in {'both', 'full'}:
        print("\nFull mode requested, seeding transformer from metaweights...")
        model.init_from_metaweights(meta)

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

    if args.continuation_mode in {'both', 'meta'}:
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

    if args.continuation_mode in {'both', 'full'}:
        # Also show the full transformer + Dario field mode for first prompt
        print("\n" + "=" * 60)
        print("  FULL MODE: transformer + Dario field (both attentions)")
        print("=" * 60)

        for prompt in prompts:
            result = continue_phrase(prompt, tokenizer, meta, model,
                                    max_tokens=30, temperature=0.45, mode='full')
            prompt_len = len(prompt)
            continuation = result[prompt_len:].strip()
            print(f"\n  prompt:       \"{prompt}\"")
            print(f"  continuation: \"{continuation[:300]}\"")

    print("\n" + "=" * 60)
    print("  PostGPT complete. The metaweights remember.")
    print("  Try: python postgpt.py \"your prompt here\"")
    print("=" * 60)


if __name__ == '__main__':
    main()
