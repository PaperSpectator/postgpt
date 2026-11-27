#pragma once
#include <regex>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <map>
#include <ranges>
#include <random>
#include <tuple>
#include <optional>
#include <vector>
#include <string>
#include <variant>

#include "json.hpp"
#include "common.hpp"

using nlohmann::json;
using namespace std;

class BPETokenizer {
    public:
        size_t max_merges;
        uint32_t vocab_size;
        bool deprioritize_merges = false;
        vector<trigram_key_t> merges;
        map<uint32_t, string> vocab;
        map<string, uint32_t> str_to_id;
        BPETokenizer(size_t max_merges=1792) {
            this->max_merges = max_merges;
            for (uint32_t i = 0; i < 256; i++) {
                this->vocab[i] = { (char)i };
                this->str_to_id[{ (char)i }] = i;
            }
        }
        vector<uint32_t> learn(string text, optional<size_t> num_merges=nullopt) {
            vector<uint32_t> ids;
            for (auto c: text) {
                ids.push_back(c);
            }
            cout << "BPE started\n";
            size_t n_merges = num_merges.value_or(this->max_merges);
            n_merges = min(n_merges, this->max_merges);
            for (size_t m = 0; m < n_merges; m++) {
                map<bigram_key_t, size_t, hash_bigram> counts = this->count_pairs(ids);
                if (counts.size() == 0) {
                    break;
                }
                bigram_key_t best_pair;
                size_t max_count = 0;
                for (const auto& key : std::views::keys(counts)) {
                    if (counts[key] > max_count) {
                        max_count = counts[key];
                        best_pair = key;
                    }
                }
                if (max_count < 2) {
                    break;
                }
                uint32_t new_id = 256 + static_cast<uint32_t>(m);
                ids = this->merge_pair(ids, best_pair, new_id);
                trigram_key_t merge;
                this->merges.push_back(trigram_key_t(get<0>(best_pair), get<1>(best_pair), new_id));
                ostringstream ss;
                ss << this->vocab[get<0>(best_pair)];
                ss << this->vocab[get<1>(best_pair)];
                string new_tok_str = ss.str();
                this->str_to_id[new_tok_str] = new_id;
                this->vocab[new_id] = new_tok_str;
            }
            cout << "BPE done\n";

            return ids;
        }
        vector<uint32_t> encode(string text) {
            vector<uint32_t> ids;
            if (this->deprioritize_merges) {
                // TODO: support regex later
                vector<string> substrs;
                // actually save preprocessors for later
                regex re("\\s+");
                sregex_token_iterator first{text.begin(), text.end(), re, -1};
                sregex_token_iterator last;
                for (; first != last; ++first) {
                    string sub_text = *first;
                    if (this->str_to_id.contains(sub_text)) {
                        ids.push_back(str_to_id[sub_text]);
                    } else {
                        for (auto c: sub_text) {
                            ids.push_back(str_to_id[{c}]);
                        }
                    }
                }
            } else {
                for (auto c: text) {
                    ids.push_back(c);
                }
            }
            for (auto m: this->merges) {
                ids = this->merge_pair(ids, bigram_key_t(get<0>(m), get<1>(m)), get<2>(m));
            }
            return ids;
        }
        string decode(vector<uint32_t> tokens) {
            ostringstream ss;
            // TODO: with deprioritize_merges, output has no spaces
            for (auto token : tokens) {
                ss << this->vocab[token];
            }
            return ss.str();
        }
        void load(string path) {
            cout << "Loading existing BPE...\n";
            ifstream f(path);
            json data = json::parse(f);
            // TODO: doesn't apply pretokenizers yet
            for (const auto& [key, val]: data["model"]["vocab"].items()) {
                this->vocab[val.get<uint32_t>()] = key;
                this->str_to_id[key] = val.get<uint32_t>();
            }
            this->vocab_size = static_cast<uint32_t>(this->vocab.size());
            if (data["model"].contains("ignore_merges")) {
                this->deprioritize_merges = data["model"]["ignore_merges"].get<bool>();
            } else {
                this->deprioritize_merges = false;
            }
            for (auto pair_toks: data["model"]["merges"]) {
                ostringstream ss;
                string tok1_string = pair_toks[0].get<string>();
                string tok2_string = pair_toks[1].get<string>();
                ss << tok1_string;
                ss << tok2_string;
                string combo_str = ss.str();
                trigram_key_t merge(
                    data["model"]["vocab"][tok1_string].get<uint32_t>(),
                    data["model"]["vocab"][tok2_string].get<uint32_t>(),
                    data["model"]["vocab"][combo_str].get<uint32_t>()
                );
                this->merges.push_back(merge);
            }
            cout << "Loading complete!\n";
            return;
        }
        void save(string path) {
            return;
        }
    private:
        map<bigram_key_t, size_t, hash_bigram> count_pairs(vector<uint32_t> ids) {
            map<bigram_key_t, size_t, hash_bigram> counts;
            for (size_t i = 0; i < ids.size() - 1; i++) {
                bigram_key_t bigram_key(ids[i], ids[i + 1]);
                if (!counts.contains(bigram_key)) {
                    counts[bigram_key] = 0;
                }
                counts[bigram_key] += 1;
            }
            return counts;
        }
        vector<uint32_t> merge_pair(vector<uint32_t> ids, bigram_key_t pair, uint32_t new_id) {
            vector<uint32_t> result;
            for (size_t i = 0; i < ids.size(); i++) {
                if (i + 1 < ids.size() && ids[i] == get<0>(pair) && ids[i+1] == get<1>(pair)) {
                    result.push_back(new_id);
                    i += 1;
                } else {
                    result.push_back(ids[i]);
                }
            }
            return result;
        }
};