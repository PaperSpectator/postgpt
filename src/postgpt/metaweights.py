import ast
from dataclasses import dataclass, field as dc_field, fields as dc_fields
import time

@dataclass
class MetaWeights:
    """
    Metaweights: weights that are implied to exist, but don't.

    After BPE tokenization of a corpus, we build:
    1. Unigram frequencies — p(token)
    2. Bigram co-occurrence — p(token_j | token_i)
    3. Trigram patterns — p(token_k | token_i, token_j)
    4. Positional affinity — which tokens prefer which positions
    5. Hebbian trace — co-occurrence memory (tokens seen together)
    6. Prophecy field — given context, what tokens are expected

    These form a probability space that a transformer can use to behave
    AS IF it had trained weights, because the statistical regularities
    from the corpus create an implicit weight space.

    The metaweights are the ghost in the machine.
    """
    vocab_size: int
    context_len: int
    unigram: list|None = None
    # Bigram: p(next | prev) — sparse dict of dict
    bigram: dict = dc_field(default_factory=dict)
    # Trigram: p(next | prev2, prev1) — sparse
    trigram: dict = dc_field(default_factory=dict)
    # Positional affinity: which tokens appear at which positions
    # token -> list of position counts
    pos_affinity: dict = dc_field(default_factory=dict)
    # Hebbian trace: co-occurrence within a window
    # (tok_a, tok_b) -> strength
    hebbian: dict = dc_field(default_factory=dict)
    total: int = 0

    def __post_init__(self):
        # Unigram: p(token)
        if not self.unigram:
            self.unigram = [0.0] * self.vocab_size

    def build(self, token_ids, window=8):
        """Build metaweight space from tokenized corpus."""
        n = len(token_ids)
        self.total = n
        t0 = time.time()

        # Unigram counts
        for tid in token_ids:
            if tid < self.vocab_size:
                self.unigram[tid] += 1.0

        # Normalize unigram
        total = sum(self.unigram)
        if total > 0:
            self.unigram = [c / total for c in self.unigram]

        # Bigram counts
        for i in range(n - 1):
            a, b = token_ids[i], token_ids[i + 1]
            if a not in self.bigram:
                self.bigram[a] = {}
            self.bigram[a][b] = self.bigram[a].get(b, 0) + 1

        # Normalize bigrams
        for a in self.bigram:
            total_a = sum(self.bigram[a].values())
            if total_a > 0:
                for b in self.bigram[a]:
                    self.bigram[a][b] /= total_a

        # Trigram counts
        for i in range(n - 2):
            key = (token_ids[i], token_ids[i + 1])
            c = token_ids[i + 2]
            if key not in self.trigram:
                self.trigram[key] = {}
            self.trigram[key][c] = self.trigram[key].get(c, 0) + 1

        # Normalize trigrams
        for key in self.trigram:
            total_k = sum(self.trigram[key].values())
            if total_k > 0:
                for c in self.trigram[key]:
                    self.trigram[key][c] /= total_k

        # Positional affinity (within context windows)
        for i in range(n):
            pos = i % self.context_len
            tid = token_ids[i]
            if tid not in self.pos_affinity:
                self.pos_affinity[tid] = [0.0] * self.context_len
            self.pos_affinity[tid][pos] += 1.0

        # Normalize positional affinity
        for tid in self.pos_affinity:
            total_t = sum(self.pos_affinity[tid])
            if total_t > 0:
                self.pos_affinity[tid] = [c / total_t for c in self.pos_affinity[tid]]

        # Hebbian trace: co-occurrence within window
        # Cap to first 20K tokens for efficiency (O(n*window))
        hebb_n = min(n, 20000)
        for i in range(hebb_n):
            for j in range(max(0, i - window), min(hebb_n, i + window + 1)):
                if i == j:
                    continue
                a, b = token_ids[i], token_ids[j]
                key = (min(a, b), max(a, b))
                decay = 1.0 / (1.0 + abs(i - j))
                self.hebbian[key] = self.hebbian.get(key, 0.0) + decay

        # Normalize hebbian
        if self.hebbian:
            max_h = max(self.hebbian.values())
            if max_h > 0:
                for key in self.hebbian:
                    self.hebbian[key] /= max_h

        elapsed = time.time() - t0
        print(f"  metaweights built: {n} tokens, {len(self.bigram)} bigram keys, "
              f"{len(self.trigram)} trigram keys, {len(self.hebbian)} hebbian pairs [{elapsed:.1f}s]")

    def query_bigram(self, prev_token, vocab_size):
        """Get bigram probability distribution given previous token."""
        dist = [1e-10] * vocab_size  # smoothing
        if prev_token in self.bigram:
            for tok, prob in self.bigram[prev_token].items():
                if tok < vocab_size:
                    dist[tok] = prob
        return dist

    def query_trigram(self, prev2, prev1, vocab_size):
        """Get trigram probability distribution given two previous tokens."""
        dist = [1e-10] * vocab_size
        key = (prev2, prev1)
        if key in self.trigram:
            for tok, prob in self.trigram[key].items():
                if tok < vocab_size:
                    dist[tok] = prob
        return dist

    def query_hebbian(self, context_tokens, vocab_size):
        """Get Hebbian resonance signal for each candidate token given context."""
        signal = [0.0] * vocab_size
        # Use sparse lookup: iterate over stored hebbian pairs only
        for (a, b), strength in self.hebbian.items():
            for ctx_tok in context_tokens:
                if a == ctx_tok and b < vocab_size:
                    signal[b] += strength
                elif b == ctx_tok and a < vocab_size:
                    signal[a] += strength
        # Normalize
        max_s = max(signal) if signal else 1.0
        if max_s > 0:
            signal = [s / max_s for s in signal]
        return signal

    def query_prophecy(self, context_tokens, vocab_size, top_k=16):
        """
        Prophecy field: given context, which tokens are expected but haven't appeared?
        Returns signal boosting tokens that "should" come next based on co-occurrence.
        """
        appeared = set(context_tokens)
        signal = [0.0] * vocab_size

        for ctx_tok in context_tokens[-4:]:  # recent context
            if ctx_tok in self.bigram:
                for tok, prob in sorted(self.bigram[ctx_tok].items(),
                                        key=lambda x: -x[1])[:top_k]:
                    if tok not in appeared and tok < vocab_size:
                        signal[tok] += prob

        max_s = max(signal) if signal else 1.0
        if max_s > 0:
            signal = [s / max_s for s in signal]
        return signal

    def fromdict(self, input_dict):
        whitelist = set([f.name for f in dc_fields(self)])
        vocab_size: int
        context_len: int
        unigram: list|None = None
        # Bigram: p(next | prev) — sparse dict of dict
        bigram: dict = dc_field(default_factory=dict)
        # Trigram: p(next | prev2, prev1) — sparse
        trigram: dict = dc_field(default_factory=dict)
        # Positional affinity: which tokens appear at which positions
        # token -> list of position counts
        pos_affinity: dict = dc_field(default_factory=dict)
        # Hebbian trace: co-occurrence within a window
        # (tok_a, tok_b) -> strength
        hebbian: dict = dc_field(default_factory=dict)
        total: int = 0
        for k, v in input_dict.items():
            if k == 'trigram':
                setattr(self, k, {ast.literal_eval(kk): vv for kk, vv in v.items()})
            elif k == 'hebbian':
                setattr(self, k, {ast.literal_eval(kk): vv for kk, vv in v.items()})
            else:
                setattr(self, k, v)

    def todict(self):
        stub = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        del stub['trigram']
        del stub['hebbian']
        stub['trigram'] = {repr(k): v for k, v in self.trigram.items()}
        stub['hebbian'] = {repr(k): v for k, v in self.hebbian.items()}
        return stub