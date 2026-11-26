
import math
import random

from ..autograd import _randn, _matrix, _zeros, linear, softmax, softmax_float, rmsnorm, Val

class PostGPT:
    """
    PostGPT: a dual-attention BPE transformer with metaweights.

    Architecture:
    - BPE tokenizer (learned from corpus)
    - Token + Position embeddings
    - N transformer blocks, each with:
        * RMSNorm
        * Dual attention: Content heads (QK^T) + RRPRAM heads (x @ Wr)
        * Residual connection
        * RMSNorm
        * MLP (expand -> ReLU -> contract)
        * Residual connection
    - Final RMSNorm -> LM head -> logits
    - Metaweight overlay: Hebbian + Prophecy + Destiny signals

    The metaweight overlay means: even with random weights, the model
    generates coherent text because the probability space from the
    corpus guides sampling through the Dario field.
    """

    def __init__(self, vocab_size, context_len=64, n_embd=48, n_head=4,
                 n_layer=2, n_content_heads=2, n_rrpram_heads=2):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_content = n_content_heads
        self.n_rrpram = n_rrpram_heads
        self.head_dim = n_embd // n_head

        assert n_content_heads + n_rrpram_heads == n_head, \
            "content + rrpram heads must equal total heads"

        # Embeddings
        self.wte = _matrix(vocab_size, n_embd)  # token embedding
        self.wpe = _matrix(context_len, n_embd)  # position embedding

        # Per-layer weights
        self.layers = []
        hd = self.head_dim
        for _ in range(n_layer):
            layer = {
                # Content attention: Q, K, V for content heads
                'wq': _matrix(n_content_heads * hd, n_embd, std=0.02),
                'wk': _matrix(n_content_heads * hd, n_embd, std=0.02),
                'wv_content': _matrix(n_content_heads * hd, n_embd, std=0.02),

                # RRPRAM attention: Wr (positional pattern matrix) + V
                'wr': _matrix(n_rrpram_heads * n_embd, context_len, std=0.02),
                'wv_rrpram': _matrix(n_rrpram_heads * hd, n_embd, std=0.02),

                # Output projection
                'wo': _matrix(n_embd, n_embd, std=0.02 / math.sqrt(2 * n_layer)),

                # MLP
                'mlp_up': _matrix(4 * n_embd, n_embd, std=0.02),
                'mlp_down': _matrix(n_embd, 4 * n_embd, std=0.02 / math.sqrt(2 * n_layer)),
            }
            self.layers.append(layer)

        # LM head
        self.lm_head = _matrix(vocab_size, n_embd, std=0.02)

        # Dario field coefficients (metaweight blending)
        self.alpha_hebbian = 0.3   # Hebbian trace strength
        self.beta_prophecy = 0.2   # Prophecy field strength
        self.gamma_destiny = 0.15  # Destiny vector strength
        self.temperature = 0.85    # Sampling temperature

        # Destiny vector (EMA of token embeddings)
        self.destiny = [0.0] * n_embd

        # Trauma accumulator
        self.trauma = 0.0

        # Collect all parameters
        self.params = []
        for row in self.wte:
            self.params.extend(row)
        for row in self.wpe:
            self.params.extend(row)
        for layer in self.layers:
            for key in layer:
                for row in layer[key]:
                    self.params.extend(row)
        for row in self.lm_head:
            self.params.extend(row)

        n_params = len(self.params)
        print(f"  PostGPT initialized: {n_params} parameters, vocab={vocab_size}, "
              f"ctx={context_len}, embd={n_embd}, heads={n_head} "
              f"(content={n_content_heads}, rrpram={n_rrpram_heads}), layers={n_layer}")

    def init_from_metaweights(self, meta):
        """
        The ghost becomes flesh.

        Instead of random initialization, seed transformer weights FROM the
        metaweight probability space. The transformer doesn't start blind —
        it starts knowing the corpus through its bones.

        1. Token embeddings ← Hebbian co-occurrence (tokens that appear together → close vectors)
        2. Position embeddings ← positional affinity (what tokens prefer which positions)
        3. RRPRAM Wr ← positional affinity patterns (the rhythm of the corpus)
        4. LM head ← unigram + bigram signal (most likely next tokens)
        """
        V = self.vocab_size
        E = self.n_embd
        T = self.context_len
        scale = 0.15  # how much metaweight signal vs random noise

        print("  Seeding transformer from metaweights (ghost → flesh)...")

        # 1. Token embeddings: tokens with high co-occurrence → similar embeddings
        #    Use SVD-free approach: for each token, its embedding is a weighted sum
        #    of its Hebbian neighbors' random embeddings
        for tok_a in range(min(V, len(self.wte))):
            signal = [0.0] * E
            n_neighbors = 0
            for tok_b in range(min(V, len(self.wte))):
                key = (min(tok_a, tok_b), max(tok_a, tok_b))
                if key in meta.hebbian and meta.hebbian[key] > 0.01:
                    strength = meta.hebbian[key]
                    for d in range(E):
                        signal[d] += strength * self.wte[tok_b][d].data
                    n_neighbors += 1
            if n_neighbors > 0:
                for d in range(E):
                    self.wte[tok_a][d].data += scale * signal[d] / n_neighbors

        # 2. Position embeddings: from positional affinity
        #    Positions that attract similar tokens → similar embeddings
        for pos in range(min(T, len(self.wpe))):
            signal = [0.0] * E
            n_toks = 0
            for tok in meta.pos_affinity:
                if tok < V and pos < len(meta.pos_affinity[tok]):
                    affinity = meta.pos_affinity[tok][pos]
                    if affinity > 0.001:
                        for d in range(E):
                            signal[d] += affinity * self.wte[tok][d].data
                        n_toks += 1
            if n_toks > 0:
                for d in range(E):
                    self.wpe[pos][d].data += scale * signal[d] / n_toks

        # 3. RRPRAM Wr: seed from positional affinity patterns
        #    Each head's Wr column[t] gets signal from which tokens prefer position t
        for layer in self.layers:
            wr = layer['wr']
            for h in range(self.n_rrpram):
                for tok in meta.pos_affinity:
                    if tok >= V:
                        continue
                    affs = meta.pos_affinity[tok]
                    for pos in range(min(T, len(affs))):
                        if affs[pos] > 0.001:
                            wr_row = h * E + (tok % E)
                            if wr_row < len(wr) and pos < len(wr[wr_row]):
                                wr[wr_row][pos].data += scale * 0.5 * affs[pos]

        # 4. LM head: seed from unigram frequencies
        #    Tokens that appear more often get higher initial bias
        for tok in range(min(V, len(self.lm_head))):
            freq = meta.unigram[tok] if tok < len(meta.unigram) else 0
            if freq > 0:
                # Spread frequency signal across embedding dimensions
                for d in range(E):
                    self.lm_head[tok][d].data += scale * freq * self.wte[tok][d].data

        print("  Metaweight seeding complete. The weights remember what they never learned.")

    def forward_token(self, token_id, pos_id, kv_cache):
        """
        Forward pass for a single token position.
        kv_cache: list of (k_list, vc_list, vr_list) per layer
        Returns logits [vocab_size] as list of Val.
        """
        hd = self.head_dim
        nc = self.n_content
        nr = self.n_rrpram

        # Token + position embedding
        tok_emb = self.wte[token_id]
        pos_emb = self.wpe[pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]

        for li in range(self.n_layer):
            layer = self.layers[li]
            k_cache, vc_cache, vr_cache = kv_cache[li]

            # Pre-norm
            x_res = x
            x_norm = rmsnorm(x)

            # ── Projections ──
            q = linear(x_norm, layer['wq'])
            k = linear(x_norm, layer['wk'])
            v_content = linear(x_norm, layer['wv_content'])
            v_rrpram = linear(x_norm, layer['wv_rrpram'])

            # Cache current position
            k_cache.append(k)
            vc_cache.append(v_content)
            vr_cache.append(v_rrpram)

            x_attn = []

            # Content heads
            for h in range(nc):
                hs = h * hd
                q_h = q[hs:hs + hd]
                k_all = [ki[hs:hs + hd] for ki in k_cache]
                v_all = [vi[hs:hs + hd] for vi in vc_cache]

                # QK^T / sqrt(d)
                attn_logits = []
                for t in range(len(k_all)):
                    score = sum(q_h[j] * k_all[t][j] for j in range(hd))
                    score = score * (1.0 / math.sqrt(hd))
                    attn_logits.append(score)

                attn_weights = softmax(attn_logits)

                head_out = []
                for j in range(hd):
                    val = sum(attn_weights[t] * v_all[t][j] for t in range(len(v_all)))
                    head_out.append(val)
                x_attn.extend(head_out)

            # ── RRPRAM attention (x @ Wr — positional pattern recognition) ──
            for h in range(nr):
                hs = h * hd
                # RRPRAM: project input through Wr to get attention over positions
                # Wr shape per head: [n_embd, context_len]
                wr_offset = h * self.n_embd
                wr_h = layer['wr'][wr_offset:wr_offset + self.n_embd]

                # x_norm @ Wr_h gives [context_len] attention pattern
                seq_len = len(k_cache)
                attn_logits = []
                for t in range(seq_len):
                    # Sum over embedding dimension
                    score = Val(0.0)
                    for d in range(min(self.n_embd, len(wr_h))):
                        if t < len(wr_h[d]):
                            score = score + x_norm[d] * wr_h[d][t]
                    attn_logits.append(score)

                # Causal mask already satisfied (we only have positions <= current)
                attn_weights = softmax(attn_logits) if attn_logits else []

                v_all = [vi[hs:hs + hd] for vi in vr_cache]
                head_out = []
                for j in range(hd):
                    val_sum = Val(0.0)
                    for t in range(len(attn_weights)):
                        if t < len(v_all):
                            val_sum = val_sum + attn_weights[t] * v_all[t][j]
                    head_out.append(val_sum)
                x_attn.extend(head_out)

            # Output projection + residual
            x_proj = linear(x_attn, layer['wo'])
            x = [a + b for a, b in zip(x_proj, x_res)]

            # MLP block
            x_res = x
            x_norm = rmsnorm(x)
            h_mlp = linear(x_norm, layer['mlp_up'])
            h_mlp = [hi.relu() for hi in h_mlp]
            x_mlp = linear(h_mlp, layer['mlp_down'])
            x = [a + b for a, b in zip(x_mlp, x_res)]

        # Final norm + LM head
        x = rmsnorm(x)
        logits = linear(x, self.lm_head)
        return logits

    def forward_sequence(self, token_ids):
        """Forward pass over a sequence. Returns list of logits per position."""
        kv_cache = [([], [], []) for _ in range(self.n_layer)]
        all_logits = []
        for pos, tid in enumerate(token_ids):
            if pos >= self.context_len:
                break
            logits = self.forward_token(tid, pos, kv_cache)
            all_logits.append(logits)
        return all_logits

    def generate(self, prompt_ids, max_tokens=64, meta=None, temperature=None):
        """
        Generate tokens autoregressively.
        If meta (MetaWeights) is provided, applies the Dario field overlay.
        """
        if temperature is None:
            temperature = self.temperature

        kv_cache = [([], [], []) for _ in range(self.n_layer)]
        generated = list(prompt_ids)
        context = list(prompt_ids)

        # Feed prompt through
        for pos, tid in enumerate(prompt_ids):
            if pos >= self.context_len - 1:
                break
            _ = self.forward_token(tid, pos, kv_cache)

        # Generate new tokens
        for step in range(max_tokens):
            pos = len(context) - 1
            if pos >= self.context_len - 1:
                break

            last_tid = context[-1]
            logits = self.forward_token(last_tid, pos, kv_cache)

            # Extract raw logit values
            raw_logits = [l.data for l in logits]

            # ── Dario Field: metaweight overlay ──
            if meta is not None:
                # Hebbian signal
                hebbian = meta.query_hebbian(context[-8:], self.vocab_size)

                # Prophecy signal
                prophecy = meta.query_prophecy(context[-8:], self.vocab_size)

                # Bigram signal
                bigram = meta.query_bigram(last_tid, self.vocab_size)

                # Trigram signal (if enough context)
                if len(context) >= 2:
                    trigram = meta.query_trigram(context[-2], context[-1], self.vocab_size)
                else:
                    trigram = [0.0] * self.vocab_size

                # Destiny update
                if last_tid < len(self.wte):
                    for d in range(self.n_embd):
                        self.destiny[d] = 0.9 * self.destiny[d] + 0.1 * self.wte[last_tid][d].data

                # Destiny signal: cosine similarity with each token embedding
                destiny_signal = [0.0] * self.vocab_size
                dest_norm = math.sqrt(sum(d * d for d in self.destiny) + 1e-10)
                if dest_norm > 1e-8:
                    for tid_c in range(min(self.vocab_size, len(self.wte))):
                        emb = [self.wte[tid_c][d].data for d in range(self.n_embd)]
                        emb_norm = math.sqrt(sum(e * e for e in emb) + 1e-10)
                        if emb_norm > 1e-8:
                            dot = sum(self.destiny[d] * emb[d] for d in range(self.n_embd))
                            destiny_signal[tid_c] = dot / (dest_norm * emb_norm)

                # Combine: Dario Equation (Leo-style: bigram DOMINATES, 12× coefficient)
                # p(x|Φ) = softmax((B_coeff·B + α·H + β·F + γ·A + trigram) / τ)
                # Metaweight signals dominate over untrained base logits
                for i in range(self.vocab_size):
                    raw_logits[i] += (self.alpha_hebbian * hebbian[i]
                                      + self.beta_prophecy * prophecy[i]
                                      + self.gamma_destiny * destiny_signal[i]
                                      + 12.0 * bigram[i]
                                      + 8.0 * trigram[i])

                # Trauma modulation
                trauma_mod = 1.0 / (1.0 + self.trauma)
                raw_logits = [l * trauma_mod for l in raw_logits]

            # Repetition penalty (Leo-style)
            recent = context[-12:] if len(context) >= 12 else context
            for t in recent:
                if t < self.vocab_size:
                    raw_logits[t] *= 0.5

            # Top-k filtering (keep top 15, mask rest)
            top_k = 15
            indexed = sorted(enumerate(raw_logits), key=lambda x: -x[1])
            threshold = indexed[min(top_k - 1, len(indexed) - 1)][1]
            for i in range(self.vocab_size):
                if raw_logits[i] < threshold:
                    raw_logits[i] = -1e10

            # Temperature + softmax
            scaled = [l / temperature for l in raw_logits]
            probs = softmax_float(scaled)

            # Sample
            r = random.random()
            cum = 0.0
            chosen = 0
            for i, p in enumerate(probs):
                cum += p
                if cum > r:
                    chosen = i
                    break

            generated.append(chosen)
            context.append(chosen)

        return generated

    def generate_meta(self, prompt_ids, max_tokens=128, meta=None, temperature=None):
        """
        Meta-generation: pure metaweight generation without transformer forward pass.
        Uses only the statistical probability space from BPE tokenization.

        This follows the Haze/Leo pattern:
        - Trigram first (most coherent), fallback to bigram, then unigram
        - Sample ONLY from tokens that actually appear in the statistics
        - Repetition penalty for loop avoidance
        - Top-k filtering (keep top 15 candidates like Leo)
        """
        if meta is None:
            return prompt_ids
        if temperature is None:
            temperature = self.temperature

        generated = list(prompt_ids)

        for _ in range(max_tokens):
            last = generated[-1]
            candidates = {}  # token_id -> count (sparse, only real candidates)

            # Try trigram first (strongest signal, like Haze)
            if len(generated) >= 2:
                key = (generated[-2], generated[-1])
                if key in meta.trigram:
                    candidates = dict(meta.trigram[key])

            # Fallback to bigram
            if not candidates and last in meta.bigram:
                candidates = dict(meta.bigram[last])

            # Fallback to unigram (last resort)
            if not candidates:
                for i in range(self.vocab_size):
                    if meta.unigram[i] > 1e-8:
                        candidates[i] = meta.unigram[i]

            if not candidates:
                break

            # Hebbian boost — contextual reinforcement on top of trigram/bigram
            ctx = generated[-4:]
            for tok in list(candidates.keys()):
                for ct in ctx:
                    key = (min(tok, ct), max(tok, ct))
                    if key in meta.hebbian:
                        candidates[tok] *= (1.0 + 0.3 * meta.hebbian[key])

            # Repetition penalty (Leo-style: penalize recently seen tokens)
            recent = generated[-12:] if len(generated) >= 12 else generated
            recent_counts = {}
            for t in recent:
                recent_counts[t] = recent_counts.get(t, 0) + 1
            for tok in list(candidates.keys()):
                if tok in recent_counts:
                    freq = recent_counts[tok]
                    penalty = 1.0 / (1.0 + 0.5 * freq)
                    candidates[tok] *= penalty

            # Top-k filtering (keep top 15, like Leo)
            top_k = 15
            sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
            sorted_cands = sorted_cands[:top_k]

            # Convert to probabilities with temperature
            tokens = [t for t, _ in sorted_cands]
            counts = [c for _, c in sorted_cands]

            # Log-space temperature scaling (like Haze SubwordField)
            import math as _math
            log_counts = [_math.log(c + 1e-10) / temperature for c in counts]
            max_lc = max(log_counts)
            exps = [_math.exp(lc - max_lc) for lc in log_counts]
            total = sum(exps)
            probs = [e / total for e in exps]

            # Sample
            r = random.random()
            cum = 0.0
            chosen = tokens[0]
            for tok, p in zip(tokens, probs):
                cum += p
                if cum > r:
                    chosen = tok
                    break

            generated.append(chosen)

        return generated