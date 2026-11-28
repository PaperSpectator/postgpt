
import struct
import time

class RawBPETokenizer:
    """Byte-Pair Encoding tokenizer. Starts with 256 byte tokens, learns merges."""

    def __init__(self, max_merges=1792):
        self.max_merges = max_merges
        self.merges = []  # list of (a, b, new_id)
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def _count_pairs(self, ids):
        """Count consecutive pairs in token list."""
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        """Replace all occurrences of pair with new_id."""
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def learn(self, data_bytes, num_merges=None):
        """Learn BPE merges from raw bytes."""
        if num_merges is None:
            num_merges = self.max_merges
        num_merges = min(num_merges, self.max_merges)

        ids = list(data_bytes)
        t0 = time.time()

        for m in range(num_merges):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break

            new_id = 256 + m
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges.append((best_pair[0], best_pair[1], new_id))
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab_size = 256 + m + 1

            if (m + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  merge {m+1}/{num_merges}  vocab={self.vocab_size}  tokens={len(ids)}  [{elapsed:.1f}s]")

        print(f"  BPE complete: {len(self.merges)} merges, vocab={self.vocab_size}, "
              f"tokens={len(ids)} (from {len(data_bytes)} bytes)")
        return ids

    def encode(self, text):
        """Encode text to token ids using learned merges."""
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    def decode(self, ids):
        """Decode token ids back to string."""
        raw = b''
        for tid in ids:
            if tid in self.vocab:
                raw += self.vocab[tid]
        return raw.decode('utf-8', errors='replace')

    def save(self, path):
        """Save merge rules to binary file."""
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', len(self.merges)))
            for a, b, new_id in self.merges:
                f.write(struct.pack('<III', a, b, new_id))

    def load(self, path):
        """Load merge rules from binary file."""
        with open(path, 'rb') as f:
            n = struct.unpack('<I', f.read(4))[0]
            self.merges = []
            for _ in range(n):
                a, b, new_id = struct.unpack('<III', f.read(12))
                self.merges.append((a, b, new_id))
                self.vocab[new_id] = self.vocab.get(a, bytes([a % 256])) + self.vocab.get(b, bytes([b % 256]))
            self.vocab_size = 256 + len(self.merges)