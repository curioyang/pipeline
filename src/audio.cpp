//
// Created by curio on 2025/2/9.
//
#include "audio.h"
#include "utils.h"
#include "wav2wav.h"

void pad_or_trim(std::vector<float>& audio, int length, int axis)
{
    // cpp中默认是vector, axis在这里没有作用
    if (audio.size() > length)
    {
        // TODO: numpy.take(indices = range(length), axis=axis)
        audio.resize(length);
        throw std::runtime_error("Audio is too long, need do more process");
    }
    else if (audio.size() < length)
    {
        int num_padding = length - audio.size();
        audio.insert(audio.end(), num_padding, 0.0f);
    }
}

std::vector<double> hann_window(int window_length, bool periodic)
{
    int N = window_length;
    if (periodic)
    {
        N = window_length + 1; // 如果是周期性窗口，窗口长度加1
    }
    auto double_PI = 2 * M_PI;
    std::vector<double> window(window_length);
    for (int n = 0; n < window_length; ++n)
    {
        // TODO: RVV 优化  (计算量有限,收益可能不明显)
        float normalized_n = static_cast<float>(n) / (N - 1.0f);
        window[n] = 0.5 - 0.5 * std::cos(double_PI * normalized_n);
        // 或者使用等效的公式：
        //         window[n] = pow(sin(M_PI * normalized_n), 2);
    }
    //    writeFloatVectorToFile(window, "/home/curio/mini-omni2-pipeline-onnxruntime/pipeline/data/window.txt");
    return window;
}

// std::vector<std::vector<std::complex<double>>>
// stft(std::vector<float> &signal, int windowSize, int hopSize, const std::vector<double> &window) {
//     // TODO: 实现需要确认
//     // pad signal
//     int pad = windowSize / 2;
//     signal = reflectPad(signal, pad, pad);
//
//     int signalLength = signal.size();
//     int numWindows = (signalLength - windowSize) / hopSize + 1;
//
//     std::vector<std::vector<std::complex<double>>> stftResult(numWindows,
//                                                               std::vector<std::complex<double>>(windowSize / 2 + 1));
//
//     // 分配 FFTW 输入和输出数组
//     double *in = (double *) fftw_malloc(sizeof(double) * windowSize);
//     fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (windowSize / 2 + 1));
//
//     // 创建 FFTW 计划
//     fftw_plan plan = fftw_plan_dft_r2c_1d(windowSize, in, out, FFTW_ESTIMATE);
//
//     for (int i = 0; i < numWindows; ++i) {
//         int start = i * hopSize;
//         std::vector<float> windowedSignal(signal.begin() + start, signal.begin() + start + windowSize);
//
//         // 应用窗口函数
//         int N = windowedSignal.size();
//         for (int ii = 0; ii < N; ++ii) {
//             windowedSignal[ii] *= window[ii];
//         }
//
//         // 复制数据到 FFTW 输入数组
//         for (int j = 0; j < windowSize; ++j) {
//             in[j] = windowedSignal[j];
//         }
//
//         // 执行 FFT
//         fftw_execute(plan);
//
//         // 存储结果
//         for (int j = 0; j < windowSize / 2 + 1; ++j) {
//             stftResult[i][j] = std::complex<double>(out[j][0], out[j][1]);
//         }
//     }
//
//     // 释放资源
//     fftw_destroy_plan(plan);
//     fftw_free(in);
//     fftw_free(out);
//     stftResult = transpose(stftResult);
//     return stftResult;
// }
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
template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix)
{
    // 获取原矩阵的行数和列数
    size_t rows = matrix.size();
    if (rows == 0) return {}; // 如果原矩阵为空，直接返回空矩阵
    size_t cols = matrix[0].size();

    // 创建一个新的矩阵，其行数为原矩阵的列数，列数为原矩阵的行数
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    // 对于每个元素进行转置操作
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}


tensor_info<float> log_mel_spectrogram(std::vector<float>& audio, int n_mels, int padding)
{
    auto window = hann_window(N_FFT);
    // auto stft_result = stft2(audio, N_FFT, HOP_LENGTH, window);
    auto stft_result = librosa::Feature::stft(audio, N_FFT, HOP_LENGTH, "hann", true, "reflect");
    stft_result = transpose(stft_result);
    std::cout << stft_result.size() << "  " << stft_result[0].size() << std::endl;

    // mag.. = stft[..., :-1].abs() ** 2
    std::vector<std::vector<float>> magnitudes(stft_result.size(), std::vector<float>(stft_result[0].size() - 1, 0));
    for (int i = 0; i < stft_result.size(); ++i)
    {
        for (int j = 0; j < stft_result[0].size() - 1; ++j)
        {
            magnitudes[i][j] = (float)std::norm(stft_result[i][j]);
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
    for (int i = 0; i < mel_filters_info.shape[0]; ++i)
    {
        for (int j = 0; j < mel_filters_info.shape[1]; ++j)
        {
            mel_filter_80[i][j] = filter_data_pointer[i * mel_filters_info.shape[1] + j];
        }
    }

    // matmul: mel_filter_80: 80*201 ; magnitudes: 201*3000
    // clmap min to 1e-10, and calculate log10
    //log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    auto mel_spec = matmul(mel_filter_80, magnitudes);
    float max_mel_spec = -INFINITY;
    for (int i = 0; i < mel_spec.size(); ++i)
    {
        for (int j = 0; j < mel_spec[0].size(); ++j)
        {
            mel_spec[i][j] = std::log10(std::max(1e-10f, mel_spec[i][j]));
            if (mel_spec[i][j] > max_mel_spec)
            {
                max_mel_spec = mel_spec[i][j];
            }
        }
    }
    tensor_info<float>mel_spec_tensor;
    mel_spec_tensor.data = std::vector<float>(mel_spec.size()*mel_spec[0].size());
    mel_spec_tensor.shape = {(long)mel_spec.size(), (long)mel_spec[0].size()};
    for (int i = 0; i < mel_spec_tensor.shape[0]; ++i)
    {
        for (int j = 0; j < mel_spec_tensor.shape[1]; ++j)
        {
            mel_spec_tensor.data[i* mel_spec_tensor.shape[1] + j] = (std::max(mel_spec[i][j], max_mel_spec - 8.0f) + 4.f) / 4.f;
        }
    }
    return std::move(mel_spec_tensor);
}

#if VAD_ENABLE
std::pair<tensor_info<float>, int> load_audio(std::vector<float> &audio, int sr)
{
    size_t frame_count = audio.size();
    auto duration_ms = (float)frame_count / sr * 1000.0f;
    pad_or_trim(audio);
    auto mel = log_mel_spectrogram(audio);
    return {mel, duration_ms / 20 + 1};
}
#else
std::pair<tensor_info<float>, int> load_audio(const std::string& path, int sr)
{
    SF_INFO sf_info;
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sf_info);
    std::vector<float> audio(sf_info.frames * sf_info.channels);
    size_t frame_count = sf_readf_float(file, audio.data(), audio.size());
    sf_close(file);

    auto duration_ms = (float)frame_count / sr * 1000.0f;
    pad_or_trim(audio);
    auto mel = log_mel_spectrogram(audio);

    return {mel, duration_ms / 20 + 1};
}
#endif

void save_audio(const std::string &path, const std::vector<float> &audio, int sr)
{
    SF_INFO sf_info;
    sf_info.samplerate = sr;
    sf_info.channels = 1;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &sf_info);
    sf_writef_float(file, audio.data(), audio.size());
}

void playAudio(const std::vector<float> &audio_hat, int sampleRate)
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        // 处理初始化错误
        return;
    }

    PlaybackData playbackData{
        audio_hat.data(),
        static_cast<unsigned long>(audio_hat.size()),
        0};

    PaStreamParameters outputParams;
    outputParams.device = Pa_GetDefaultOutputDevice();
    if (outputParams.device == paNoDevice)
    {
        Pa_Terminate();
        return;
    }
    outputParams.channelCount = 1;         // 单声道
    outputParams.sampleFormat = paFloat32; // 32位浮点格式
    outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    PaStream *stream;
    err = Pa_OpenStream(&stream,
                        nullptr,
                        &outputParams,
                        sampleRate,
                        paFramesPerBufferUnspecified,
                        paClipOff,
                        audioCallback,
                        &playbackData);

    if (err != paNoError)
    {
        Pa_Terminate();
        return;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        Pa_CloseStream(stream);
        Pa_Terminate();
        return;
    }

    // 等待播放完成
    while (Pa_IsStreamActive(stream))
    {
        Pa_Sleep(100);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
}

static int audioCallback(const void *input, void *output,
                         unsigned long frameCount,
                         const PaStreamCallbackTimeInfo *timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void *userData)
{
    PlaybackData *data = (PlaybackData *)userData;
    float *out = (float *)output;
    unsigned long framesToCopy = frameCount;

    // 计算剩余帧数
    if (data->currentFrame + framesToCopy > data->totalFrames)
        framesToCopy = data->totalFrames - data->currentFrame;

    if (framesToCopy > 0)
    {
        // 复制数据到输出缓冲区
        memcpy(out, data->data + data->currentFrame, framesToCopy * sizeof(float));
        data->currentFrame += framesToCopy;
    }

    // 静音填充剩余缓冲区（如果有）
    if (framesToCopy < frameCount)
        memset(out + framesToCopy, 0, (frameCount - framesToCopy) * sizeof(float));

    // 判断是否播放完毕
    return (data->currentFrame >= data->totalFrames) ? paComplete : paContinue;
}
