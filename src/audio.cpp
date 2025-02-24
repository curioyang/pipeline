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
    std::vector<std::vector<float>> mel;
    {
        ScopedTiming st("melspectrogram");
        mel = librosa::Feature::melspectrogram(audio, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann",
                                                true, "reflect", 2.0f, WHISPER_N_MELS, 0.0f,
                                                WHISPER_SAMPLE_RATE / 2.0f);
    }

    std::vector<std::vector<float>> mel_spec;
    {
        ScopedTiming st("mel transpose");
        mel_spec = transpose(mel);
    }

    ScopedTiming st("mel max + log10");
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
    tensor_info<float> mel_spec_tensor;
    mel_spec_tensor.data = std::vector<float>(WHISPER_N_MELS * 3000, 0);
    mel_spec_tensor.shape = {WHISPER_N_MELS, 3000};
    int m = WHISPER_N_MELS;
    int n = mel_spec[0].size();

    for (int i = 0; i < m; ++i)
    {
        int index = i * 3000;
        for (int j = 0; j < n; ++j)
        {
            mel_spec_tensor.data[index + j] = (std::max(mel_spec[i][j], max_mel_spec - 8.0f) + 4.f) / 4.f;
        }
    }
    return std::move(mel_spec_tensor);
}

std::pair<tensor_info<float>, int> load_audio(std::vector<float> &audio, int sr)
{
    size_t frame_count = audio.size();
    auto duration_ms = (float)frame_count / sr * 1000.0f;
    // pad_or_trim(audio);
    auto mel = log_mel_spectrogram(audio);
    return {mel, duration_ms / 20 + 1};
}

// std::pair<tensor_info<float>, int> load_audio(const std::string& path, int sr)
// {
//     SF_INFO sf_info;
//     SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sf_info);
//     std::vector<float> audio(sf_info.frames * sf_info.channels);
//     size_t frame_count = sf_readf_float(file, audio.data(), audio.size());
//     sf_close(file);

//     auto duration_ms = (float)frame_count / sr * 1000.0f;
//     // pad_or_trim(audio);
//     auto mel = log_mel_spectrogram(audio);

//     return {mel, duration_ms / 20 + 1};
// }
// #endif

void save_audio(const std::string &path, const std::vector<float> &audio, int sr)
{
    // libwav
    // wav::WavWriter WW(audio.data(), audio.size(), 1, sr, 32);
    // WW.Write(path);

    // libsndfile
    SF_INFO sf_info;
    sf_info.samplerate = sr;
    sf_info.channels = 1;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &sf_info);
    sf_writef_float(file, audio.data(), audio.size());

    // audiofile
    // AudioFile.write
}

// void playAudio(const std::vector<float> &audio_hat, int sampleRate)
// {
//     PaError err = Pa_Initialize();
//     if (err != paNoError)
//     {
//         // 处理初始化错误
//         return;
//     }

//     PlaybackData playbackData{
//         audio_hat.data(),
//         static_cast<unsigned long>(audio_hat.size()),
//         0};

//     PaStreamParameters outputParams;
//     outputParams.device = Pa_GetDefaultOutputDevice();
//     if (outputParams.device == paNoDevice)
//     {
//         Pa_Terminate();
//         return;
//     }
//     outputParams.channelCount = 1;         // 单声道
//     outputParams.sampleFormat = paFloat32; // 32位浮点格式
//     outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
//     outputParams.hostApiSpecificStreamInfo = nullptr;

//     PaStream *stream;
//     err = Pa_OpenStream(&stream,
//                         nullptr,
//                         &outputParams,
//                         sampleRate,
//                         paFramesPerBufferUnspecified,
//                         paClipOff,
//                         audioCallback,
//                         &playbackData);

//     if (err != paNoError)
//     {
//         Pa_Terminate();
//         return;
//     }

//     err = Pa_StartStream(stream);
//     if (err != paNoError)
//     {
//         Pa_CloseStream(stream);
//         Pa_Terminate();
//         return;
//     }

//     // 等待播放完成
//     while (Pa_IsStreamActive(stream))
//     {
//         Pa_Sleep(100);
//     }

//     Pa_StopStream(stream);
//     Pa_CloseStream(stream);
//     Pa_Terminate();
// }

// static int audioCallback(const void *input, void *output,
//                          unsigned long frameCount,
//                          const PaStreamCallbackTimeInfo *timeInfo,
//                          PaStreamCallbackFlags statusFlags,
//                          void *userData)
// {
//     PlaybackData *data = (PlaybackData *)userData;
//     float *out = (float *)output;
//     unsigned long framesToCopy = frameCount;

//     // 计算剩余帧数
//     if (data->currentFrame + framesToCopy > data->totalFrames)
//         framesToCopy = data->totalFrames - data->currentFrame;

//     if (framesToCopy > 0)
//     {
//         // 复制数据到输出缓冲区
//         memcpy(out, data->data + data->currentFrame, framesToCopy * sizeof(float));
//         data->currentFrame += framesToCopy;
//     }

//     // 静音填充剩余缓冲区（如果有）
//     if (framesToCopy < frameCount)
//         memset(out + framesToCopy, 0, (frameCount - framesToCopy) * sizeof(float));

//     // 判断是否播放完毕
//     return (data->currentFrame >= data->totalFrames) ? paComplete : paContinue;
// }
