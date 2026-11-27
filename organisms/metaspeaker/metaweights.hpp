#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <map>
#include <map>
#include <ranges>
#include <random>
#include <tuple>
#include <optional>
#include <vector>
#include <variant>

#include "common.hpp"

using namespace std;

class MetaWeights {
    public:
        uint32_t vocab_size  = DEFAULT_VOCAB_SIZE;
        size_t context_len   = DEFAULT_CONTEXT_LEN;
        size_t tokens_seen   = 0;
        vector<float> unigram;
        map<bigram_key_t, float, hash_bigram> bigram;
        map<trigram_key_t, uint32_t, hash_trigram> trigram;
        map<uint32_t, vector<uint32_t>> pos_affinity;
        map<hebbian_key_t, float, hash_bigram> hebbian;
        // TODO: how to normalize as we collect
        MetaWeights(uint32_t vocab_size, size_t context_len) {
            this->vocab_size = vocab_size;
            this->context_len = context_len;
            unigram.assign(this->vocab_size, 0.0f);
        }
        void build(vector<uint32_t> token_ids, size_t window = 8, size_t hebbian_scan_limit=20000) {
            size_t n = token_ids.size();
            float valid_tokens_seen = 0.0;
            // collect unigrams, bigrams, and trigrams
            // maps we only use for collecting totals
            map<uint32_t, float>     bigram_total;
            map<bigram_key_t, float, hash_bigram> trigram_total;
            map<uint32_t, float>     pos_affinity_total;
            if (token_ids.size() == 0) {
                return;
            }

            for (size_t i = 0; i < token_ids.size() - 2; i++) {
                uint32_t tid         = token_ids[i];
                uint32_t tid_next    = token_ids[i + 1];
                uint32_t tid_next2   = token_ids[i + 2];
                bigram_key_t  bigram_key(tid, tid_next);
                trigram_key_t trigram_key(tid, tid_next, tid_next2);

                // collect unigram
                if (tid < vocab_size) {
                    this->unigram[tid] += 1.0;
                    valid_tokens_seen += 1.0;
                }

                // collect bigram
                if (!this->bigram.contains(bigram_key)) {
                    this->bigram[bigram_key] = 0.0;
                }
                bigram_total[tid]        += 1.0;
                this->bigram[bigram_key] += 1.0;

                // collect trigram
                if (!this->trigram.contains(trigram_key)) {
                    this->trigram[trigram_key] = 0.0;
                }
                this->trigram[trigram_key] += 1.0;
                trigram_total[bigram_key]  += 1.0;

                
                size_t ctx_pos = i % this->context_len;
                if (this->pos_affinity[tid].size() == 0) {
                    this->pos_affinity[tid].assign(this->context_len, 0.0);
                }
                this->pos_affinity[tid][ctx_pos] += 1.0;
                pos_affinity_total[tid]          += 1.0;
            }

            // collect the remaining unigrams
            bool need_remaining_bigram = true;
            for (size_t i = token_ids.size() - 2; i < token_ids.size(); i++) {
                uint32_t tid = token_ids[i];
                if (tid < vocab_size) {
                    this->unigram[tid] += 1.0;
                    valid_tokens_seen += 1.0;
                }
                if (need_remaining_bigram) {
                    uint32_t tid_next = token_ids[i] + 1;
                    bigram_key_t  bigram_key(tid, tid_next);
                    this->bigram[bigram_key] += 1.0;
                    bigram_total[tid]       += 1.0;
                    need_remaining_bigram = false;
                }
                size_t ctx_pos = i % this->context_len;
                this->pos_affinity[tid][ctx_pos] += 1.0;
            }

            // normalize unigrams
            for (size_t i = 0; i < this->unigram.size(); i++) {
                this->unigram[i] /= valid_tokens_seen;
            }

            // normalize bigrams
            for (const auto& key : std::views::keys(this->bigram)) {
                uint32_t total_key = get<0>(key);
                this->bigram[key] /= bigram_total[total_key];
            }

            // normalize trigrams
            for (const auto& key : std::views::keys(this->trigram)) {
                bigram_key_t total_key(get<0>(key), get<1>(key));
                this->trigram[key] /= trigram_total[total_key];
            }

            // normalize positional affinity
            for (const auto& key : std::views::keys(this->pos_affinity)) {
                for (size_t i = 0; i < this->pos_affinity[key].size(); i++) {
                    this->pos_affinity[key][i] /= pos_affinity_total[key];
                }
            }

            // hebbian scan
            size_t hebb_n = min(hebbian_scan_limit, token_ids.size());
            float hebbian_max = -99999999999999.f;
            for (size_t i = 0; i < hebb_n; i++) {
                for (size_t j = max(static_cast<std::size_t>(0), i - window); j < min(hebb_n, i + window + static_cast<std::size_t>(1)); j++) {
                    if (i == j) {
                        continue;
                    }
                    uint32_t a = token_ids[i];
                    uint32_t b = token_ids[j];
                    hebbian_key_t hebb_key(min(a, b), max(a, b));
                    float decay = 1.0 / (1.0 - abs(static_cast<float>(i) - static_cast<float>(j)));
                    if (!this->hebbian.contains(hebb_key)) {
                        this->hebbian[hebb_key] = 0.0;
                    }
                    this->hebbian[hebb_key] += decay;
                    if (this->hebbian[hebb_key] > hebbian_max) {
                        hebbian_max = this->hebbian[hebb_key];
                    }
                }
            }

            if (this->hebbian.size() > 0) {
                for (const auto& key : std::views::keys(this->hebbian)) {
                    this->hebbian[key] /= hebbian_max;
                }
            }
        }
        vector<uint32_t> generate(vector<uint32_t> prompt_ids, size_t max_tokens=128, float temperature=0.85, size_t window=4, size_t penalize_recent=12, optional<uint32_t> seed = nullopt, size_t top_k=15) {
            vector<uint32_t> generated(prompt_ids);
            if (prompt_ids.size() == 0) {
                return generated;
            }
            for (size_t i = 0; i < max_tokens; i++) {
                uint32_t last = generated[generated.size() - 1];
                map<uint32_t, float> candidates;
                if (generated.size() >= 2) {
                    uint32_t second_to_last = generated[generated.size() - 2];
                    bigram_key_t partial_trigram_key(second_to_last, last);
                    auto it = this->trigram.lower_bound(partial_trigram_key);
                    //while (it != data.end() && !(partial_trigram_key < it->first)) {
                    while (it != this->trigram.end() && !(partial_trigram_key < bigram_key_t(get<0>(it->first), get<1>(it->first)))) {
                        candidates[get<2>(it->first)] = it->second;
                        ++it;
                    }
                }
                if (candidates.size() == 0 && this->bigram.contains(last)) {
                    unigram_key_t partial_bigram_key(last);
                    auto it = this->bigram.lower_bound(partial_bigram_key);
                    while (it != this->bigram.end() && !(partial_bigram_key < get<0>(it->first))) {
                        candidates[get<1>(it->first)] = it->second;
                        ++it;
                    }
                }
                if (candidates.size() == 0) {
                    for (size_t j = 0; j < this->vocab_size; j++) {
                        if (this->unigram[j] > 1e-8) {
                            candidates[j] = this->unigram[j];
                        }
                    }
                }
                if (candidates.size() == 0) {
                    break;
                }

                span<uint32_t> ctx = span{generated}.last(4);
                for (const auto& tok : std::views::keys(candidates)) {
                    for (auto ct: ctx) {
                        hebbian_key_t heb_key(min(tok, ct), max(tok, ct));
                        if (this->hebbian.contains(heb_key)) {
                            candidates[tok] *= (1.0 + 0.3 * this->hebbian[heb_key]);
                        }
                    }
                }

                span<uint32_t> recent;
                map<uint32_t, float> recent_counts;
                if (generated.size() >= penalize_recent) {
                    recent = span{generated}.last(penalize_recent);
                } else {
                    recent = span{generated};
                }
                for (auto t: recent) {
                    if (!recent_counts.contains(t)) {
                        recent_counts[t] = 0.0;
                    }
                    recent_counts[t] += 1.0;
                }
                for (const auto& key : std::views::keys(candidates)) {
                    if (recent_counts.contains(key)) {
                        float freq = recent_counts[key];
                        float penalty = 1.0 / (1.0 + 0.5 * freq);
                        candidates[key] *= penalty;
                    }
                }

                vector<pair<uint32_t, float>> sorted_candidates(candidates.begin(), candidates.end());
                sort(sorted_candidates.begin(), sorted_candidates.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });
                span<pair<uint32_t, float>> top_k_window;
                if (sorted_candidates.size() < top_k) {
                    top_k_window = span{sorted_candidates};
                } else {
                    top_k_window = span{sorted_candidates}.last(top_k);
                }

                vector<float> log_counts(top_k_window.size(), 0.0);
                vector<float> exps(top_k_window.size(), 0.0);
                vector<float> probs(top_k_window.size(), 0.0);
                float total = 0.0;
                float max_lc = -999999999999999999.f;
                for (size_t i = 0; i < log_counts.size(); i++) {
                    log_counts[i] = log(top_k_window[i].second + 1e-10) / temperature;
                    if (log_counts[i] > max_lc) {
                        max_lc = log_counts[i];
                    }
                }
                for (size_t i = 0; i < exps.size(); i++) {
                    exps[i] = exp(log_counts[i] - max_lc);
                    total += exps[i];
                }
                for (size_t i = 0; i < probs.size(); i++) {
                    probs[i] = exps[i] / total;
                }
                
                mt19937 gen(seed.value_or(random_device{}()));
                uniform_real_distribution<> dis(0.0, 1.0);
                float r = dis(gen);
                float cumulative_prob = 0.0;
                uint32_t chosen = top_k_window[0].first;
                for (size_t i = 0; i < top_k_window.size(); i++) {
                    cumulative_prob += probs[i];
                    if (cumulative_prob > r) {
                        chosen = top_k_window[i].first;
                        break;
                    }
                }
                generated.push_back(chosen);
            }

            return generated;
        }
};