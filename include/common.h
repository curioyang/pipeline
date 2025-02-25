
#pragma once

#include <vector>

constexpr int hash_flag = -9999;

constexpr int padded_audio_vocab = 4096 + 64;

constexpr int text_vocabsize = 151936;
constexpr int text_specialtokens = 64;
constexpr int audio_vocabsize = 4096;
constexpr int audio_specialtokens = 64;

constexpr int padded_text_vocabsize = text_vocabsize + text_specialtokens;
constexpr int padded_audio_vocabsize = audio_vocabsize + audio_specialtokens;

constexpr int _eot = text_vocabsize;
constexpr int _pad_t = text_vocabsize + 1;
constexpr int _input_t = text_vocabsize + 2;
constexpr int _answer_t = text_vocabsize + 3;
constexpr int _asr = text_vocabsize + 4;

constexpr int _eoa = audio_vocabsize;
constexpr int _pad_a = audio_vocabsize + 1;
constexpr int _input_a = audio_vocabsize + 2;
constexpr int _answer_a = audio_vocabsize + 3;
constexpr int _split = audio_vocabsize + 4;
constexpr int _image = audio_vocabsize + 5;
constexpr int _eoimage = audio_vocabsize + 6;

constexpr int WHISPER_N_MELS = 80;
constexpr int WHISPER_SAMPLE_RATE = 16000;
constexpr int WHISPER_N_FFT = 400;
constexpr int WHISPER_HOP_LENGTH = 160;

constexpr int WHISPER_SEQ_LENGTH = 1500;

template<class T>
struct tensor_info {
    std::vector<T> data;
    std::vector<long> shape;
};
