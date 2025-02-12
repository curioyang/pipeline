//
// Created by curio on 2025/2/6.
//
#pragma once

#ifndef WAV2WAV_AUDIO_H
#define WAV2WAV_AUDIO_H
#include <cassert>
#include <complex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#if defined(ONNX)
#include <sndfile.h>
#include "librosa.h"
#include "cnpy.h"
#endif

//               utils
///////////////////////////////////////////////////
//void writeFloatVectorToFile(const std::vector<float>& vec, const std::string& filename) {
//    std::ofstream file(filename);
//    if (!file.is_open()) {
//        std::cerr << "Failed to open file for writing: " << filename << std::endl;
//        return;
//    }
//
//    for (const auto& value : vec) {
//        file << value << " ";
//    }
//
//    // 可选：添加换行符
//    file << std::endl;
//
//    file.close();
//    std::cout << "Float vector written to file: " << filename << std::endl;
//}

static int exact_div(int x, int y) {
    assert(x % y == 0);
    return x / y;
}
///////////////////////////////////////////////////

constexpr int SAMPLE_RATE = 16000;
constexpr int N_FFT = 400;
constexpr int HOP_LENGTH = 160;
constexpr int CHUNK_LENGTH = 30;
constexpr int N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE;  //# 480000 samples in a 30-second chunk;
//constexpr  int N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH);  //# 3000 frames in a mel spectrogram input;
constexpr int N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2; // # the initial convolutions has stride 2;
//constexpr  int FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH);  //# 10ms per audio frame;
//constexpr  int TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN); // # 20ms per audio token;



void pad_or_trim(std::vector<float> &audio, int length = N_SAMPLES, int axis = -1);

std::vector<double> hann_window(int window_length = N_FFT, bool periodic = true);

std::vector<std::vector<std::complex<double>>>
stft(std::vector<float> &signal, int windowSize, int hopSize, const std::vector<double> &window);
//
// typedef std::complex<double> Complex;
//
// // 计算复数指数
// Complex exp_j(double theta) {
//     return Complex(cos(theta), sin(theta));
// }
//
// std::vector<std::vector<std::complex<double>>>
// stft2(std::vector<float> &signal, int windowSize, int hopSize, const std::vector<double> &window)
// {
//     std::vector<std::vector<std::complex<double>>> X; // 存储结果
//
//     // 遍历所有频率
//     for (int omega = 0; omega < windowSize; ++omega) {
//         std::vector<Complex> frame_result(windowSize, Complex(0, 0)); // 初始化当前帧的结果
//         for (int k = 0; k < windowSize; ++k) {
//             frame_result[k] = (double)signal[k] * exp_j(-2 * M_PI * omega * k / windowSize);
//         }
//         X.push_back(frame_result);
//     }
//
//     return X;
// }

std::vector<std::vector<float>> log_mel_spectrogram(std::vector<float> &audio, int n_mels = 80, int padding = 0);

#if VAD_ENABLE
std::pair<std::vector<std::vector<float>>, int> load_audio(std::vector<float> &audio, int sr = SAMPLE_RATE);
#else
std::pair<std::vector<std::vector<float>>, int> load_audio(const std::string &path, int sr = SAMPLE_RATE);
#endif

#endif //WAV2WAV_AUDIO_H
