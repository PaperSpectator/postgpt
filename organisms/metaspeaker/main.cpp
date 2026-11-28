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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <span>

#include "RawBPETokenizer.hpp"
#include "metaweights.hpp"
#include "CLI11.hpp"

using namespace std;

int main(int argc, char **argv)
{
    RawBPETokenizer tokenizer;
    CLI::App app("metaspeaker");
    string corpus_path = "postgpt.txt";
    string raw_bpe_path    = "metaspeaker.model";
    string prompt = "What is the meaning of you?";
    size_t context_len    = DEFAULT_CONTEXT_LEN;
    uint32_t vocab_size   = DEFAULT_VOCAB_SIZE;
    bool train = true;
    app.add_option("--file", corpus_path, "The path to initialize the metaweights");
    app.add_option("--prompt", prompt, "The question you want to ask the metaspeaker");
    app.add_flag("--train,!--no-train", train, "Enable/disable BPE training before speaking");
    app.add_option("--raw-bpe", raw_bpe_path, "BPE to load");
    app.add_option("--context-len", context_len, "Context length");
    app.add_option("--vocab-size", vocab_size, "Vocab size");
    CLI11_PARSE(app, argc, argv);

    ifstream corpus(corpus_path);
    ostringstream buffer;
    buffer << corpus.rdbuf();
    string full_content = buffer.str();

    if (train) {
        tokenizer.learn(full_content);
        tokenizer.save(raw_bpe_path);
    } else {
        tokenizer.load(raw_bpe_path);
    }
    //tokenizer.load_merges(merges_path);
    vector<uint32_t> tokens(tokenizer.encode(full_content));

    MetaWeights metaspeaker(vocab_size, context_len);
    cout << "metaspeaker: " << metaspeaker.vocab_size << "\n";
    /*
    vector<uint32_t> nonexistent_corpus = {
        38,  4,  6, 19, 32,  3, 40, 27,
        1,  16, 25, 34, 35, 37, 30,  7,
        29, 39, 24, 14, 22, 31, 36,  8,
        9,  18, 21,  5, 10, 23, 17, 12,
        11, 13, 33,  2, 15, 26, 28, 20,
    };
    */
    metaspeaker.build(tokens);
    cout << "metaspeaker has seen words!\n";
    cout << "bigrams: "  << metaspeaker.bigram.size() << "\n";
    cout << "trigrams: " << metaspeaker.trigram.size() << "\n";
    cout << "hebbian: "  << metaspeaker.hebbian.size() << "\n\n";

    cout << "metaspeaker speaks now!\n";
    vector<uint32_t> prompt_tokens(tokenizer.encode(prompt));
    auto result = metaspeaker.generate(prompt_tokens);
    auto result_str = tokenizer.decode(result);
    cout << result_str << "\n";
    /*
    for (size_t i = 0; i < result.size(); i++) {
        cout << result[i];
        if (i < result.size() - 1) {
            cout << ", ";
        }
    }
    */
    cout << "\n";
}