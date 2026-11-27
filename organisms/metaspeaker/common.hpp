#pragma once
#include <tuple>
#include <variant>
#define DEFAULT_VOCAB_SIZE    2048
#define DEFAULT_CONTEXT_LEN   64

using namespace std;
using unigram_key_t = uint32_t;
using bigram_key_t = tuple<unigram_key_t, unigram_key_t>;
using trigram_key_t = tuple<unigram_key_t, unigram_key_t, unigram_key_t>;
using hebbian_key_t = tuple<unigram_key_t, unigram_key_t>;

struct hash_bigram {
    using is_transparent = void;

    bool operator() (const bigram_key_t& full, const unigram_key_t& partial) const {
        return get<0>(full) < partial;
    }
    bool operator() (const unigram_key_t& partial, const bigram_key_t& full) const {
        return  partial < get<0>(full);
    }
    bool operator() (const bigram_key_t& full, const bigram_key_t& also_full) const {
        return full < also_full;
    }

};

struct hash_trigram {
    using is_transparent = void;

    bool operator() (const trigram_key_t& full, const bigram_key_t& partial) const {
        return tie(get<0>(full), get<1>(full)) < partial;
    }
    bool operator() (const bigram_key_t& partial, const trigram_key_t& full) const {
        return partial < tie(get<0>(full), get<1>(full));
    }
    bool operator() (const trigram_key_t& full, const trigram_key_t& also_full)  const {
        return full < also_full;
    }

};