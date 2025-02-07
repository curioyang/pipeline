//
// Created by curio on 2025/2/6.
//

#ifndef WAV2WAV_AUDIO_H
#define WAV2WAV_AUDIO_H

#include <cassert>
#include <complex>
//#include "librosa.h"
#include <cmath>
#include <fstream>
#include <fftw3.h>
#include <iostream>
#include "cnpy.h"
#include "utils.h"
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



void pad_or_trim(std::vector<float> &audio, int length = N_SAMPLES, int axis = -1) {
    // cpp中默认是vector, axis在这里没有作用
    if (audio.size() > length) {
        // TODO: numpy.take(indices = range(length), axis=axis)
        audio.resize(length);
        throw std::runtime_error("Audio is too long, need do more process");
    } else if (audio.size() < length) {
        int num_padding = length - audio.size();
        audio.insert(audio.end(), num_padding, 0.0f);
    }
}

std::vector<double> hann_window(int window_length = N_FFT, bool periodic = true) {
    int N = window_length;
    if (periodic) {
        N = window_length + 1; // 如果是周期性窗口，窗口长度加1
    }
    auto double_PI = 2 * M_PI;
    std::vector<double> window(window_length);
    for (int n = 0; n < window_length; ++n) {
        // TODO: RVV 优化  (计算量有限,收益可能不明显)
        float normalized_n = static_cast<float>(n) / (N - 1.0f);
        window[n] = 0.5 - 0.5 * std::cos(double_PI * normalized_n);
        // 或者使用等效的公式：
//         window[n] = pow(sin(M_PI * normalized_n), 2);
    }
//    writeFloatVectorToFile(window, "/home/curio/mini-omni2-pipeline-onnxruntime/pipeline/data/window.txt");
    return window;
}

std::vector<std::vector<std::complex<double>>>
stft(std::vector<float> &signal, int windowSize, int hopSize, const std::vector<double> &window) {
    // TODO: 实现需要确认
    // pad signal
    int pad = windowSize / 2;
    signal = reflectPad(signal, pad, pad);

    int signalLength = signal.size();
    int numWindows = (signalLength - windowSize) / hopSize + 1;

    std::vector<std::vector<std::complex<double>>> stftResult(numWindows,
                                                              std::vector<std::complex<double>>(windowSize / 2 + 1));

    // 分配 FFTW 输入和输出数组
    double *in = (double *) fftw_malloc(sizeof(double) * windowSize);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (windowSize / 2 + 1));

    // 创建 FFTW 计划
    fftw_plan plan = fftw_plan_dft_r2c_1d(windowSize, in, out, FFTW_ESTIMATE);

    for (int i = 0; i < numWindows; ++i) {
        int start = i * hopSize;
        std::vector<float> windowedSignal(signal.begin() + start, signal.begin() + start + windowSize);

        // 应用窗口函数
        int N = windowedSignal.size();
        for (int ii = 0; ii < N; ++ii) {
            windowedSignal[ii] *= window[ii];
        }

        // 复制数据到 FFTW 输入数组
        for (int j = 0; j < windowSize; ++j) {
            in[j] = windowedSignal[j];
        }

        // 执行 FFT
        fftw_execute(plan);

        // 存储结果
        for (int j = 0; j < windowSize / 2 + 1; ++j) {
            stftResult[i][j] = std::complex<double>(out[j][0], out[j][1]);
        }
    }

    // 释放资源
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    stftResult = transpose(stftResult);
    return stftResult;
}

std::vector<std::vector<float>> log_mel_spectrogram(std::vector<float> &audio, int n_mels = 80, int padding = 0) {
    auto window = hann_window(N_FFT);
    auto stft_result = stft(audio, N_FFT, HOP_LENGTH, window);
    std::cout << stft_result.size() << "  " << stft_result[0].size() << std::endl;

    // mag.. = stft[..., :-1].abs() ** 2
    std::vector<std::vector<float>> magnitudes(stft_result.size(), std::vector<float>(stft_result[0].size() - 1, 0));
    for (int i = 0; i < stft_result.size(); ++i) {
        for (int j = 0; j < stft_result[0].size() - 1; ++j) {
            magnitudes[i][j] = (float) std::norm(stft_result[i][j]);
        }
    }

    // load mel filter
    // from librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
    // Pythonlib: whisper->audio.py->mel_filters:92L
    // Path is python3.12/site-packages/whisper/assets/mel_filters.npz : contains two arrays: mel_80 and mel_128
    // here we need mel_80.
    // TODO: remove libcnpy, use mel_80 directly. use python to convert mel_filters.npz[0] to mel_filters.bin
    auto filters = cnpy::npz_load("../data/mel_filters.npz");
    auto mel_filters_info = filters["mel_80"];
    std::vector<std::vector<float>> mel_filter_80(mel_filters_info.shape[0],
                                                  std::vector<float>(mel_filters_info.shape[1], 0));
    auto filter_data_pointer = mel_filters_info.data<float>();
    for (int i = 0; i < mel_filters_info.shape[0]; ++i) {
        for (int j = 0; j < mel_filters_info.shape[1]; ++j) {
            mel_filter_80[i][j] = filter_data_pointer[i * mel_filters_info.shape[1] + j];
        }
    }

    // matmul: mel_filter_80: 80*201 ; magnitudes: 201*3000
    // clmap min to 1e-10, and calculate log10
    //log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    auto mel_spec = matmul(mel_filter_80, magnitudes);
    for (int i = 0; i < mel_spec.size(); ++i) {
        for (int j = 0; j < mel_spec[0].size(); ++j) {
            mel_spec[i][j] = std::log10(std::max(1e-10f, mel_spec[i][j]));
            mel_spec[i][j] = (std::max(mel_spec[i][j], mel_spec[i][j] - 8.0f) + 4.f) / 4.f;
        }
    }
    return mel_spec;
}

std::pair<std::vector<std::vector<float>>, int> load_audio(const std::string &path, int sr = SAMPLE_RATE) {
    SF_INFO sf_info;
    SNDFILE *file = sf_open(path.c_str(), SFM_READ, &sf_info);
    std::vector<float> audio(sf_info.frames * sf_info.channels);
    size_t frame_count = sf_readf_float(file, audio.data(), audio.size());
    sf_close(file);
    auto duration_ms = frame_count / sr * 1000.0f;
    // 实现音频预处理（标准化、预加重、分帧等）
    pad_or_trim(audio);
    auto mel = log_mel_spectrogram(audio);


    // 生成Mel频谱（需要实现FFT和Mel滤波器组）
    // 返回Mel频谱和长度信息
//    return {mel_spectrogram, length};
    return {mel, duration_ms/20 + 1};
}

#endif //WAV2WAV_AUDIO_H
