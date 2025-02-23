// //
// // Created by curio on 2025/2/13.
// //

// #ifndef WAV2WAV_AUDIO_PLAY_H
// #define WAV2WAV_AUDIO_PLAY_H

// #include <portaudio.h>
// #include <vector>
// #include <thread>
// #include <atomic>
// #include <mutex>
// #include <condition_variable>
// #include <cmath>
// #include <cstring>

// // 线程安全的环形缓冲区
// template <typename T>
// class RingBuffer {
// public:
//     RingBuffer(size_t capacity)
//             : m_capacity(capacity),
//               m_buffer(new T[capacity]),
//               m_head(0),
//               m_tail(0),
//               m_count(0) {}

//     ~RingBuffer() {
//         delete[] m_buffer;
//     }

//     bool write(const T* data, size_t samples) {
//         std::unique_lock<std::mutex> lock(m_mutex);
//         if (samples > availableWrite()) {
//             return false; // 空间不足
//         }

//         for (size_t i = 0; i < samples; ++i) {
//             m_buffer[m_head] = data[i];
//             m_head = (m_head + 1) % m_capacity;
//         }
//         m_count += samples;
//         m_cv.notify_one();
//         return true;
//     }

//     size_t read(T* data, size_t maxSamples) {
//         std::unique_lock<std::mutex> lock(m_mutex);

//         // 等待条件：数据可用或线程停止
//         m_cv.wait(lock, [this]() {
//                 return m_count > 0 || !isRunning;
//         });

//         if(!isRunning){
//             return 0;
//         }

//         size_t samplesToRead = std::min(maxSamples, m_count);
//         for (size_t i = 0; i < samplesToRead; ++i) {
//             data[i] = m_buffer[m_tail];
//             m_tail = (m_tail + 1) % m_capacity;
//         }
//         m_count -= samplesToRead;
//         return samplesToRead;
//     }

//     size_t availableWrite() const {
//         return m_capacity - m_count;
//     }
//     bool isEmpty() const {
//         return m_count == 0;
//     }

//     size_t availableRead() const {
//         return m_count;
//     }

//     void stop() {
//         isRunning = false;
//         m_cv.notify_all();
//     }

// private:
//     T* m_buffer;
//     size_t m_capacity;
//     size_t m_head;
//     size_t m_tail;
//     size_t m_count;
//     mutable std::mutex m_mutex;
//     std::condition_variable m_cv;
//     std::atomic<bool> isRunning = true;
// };

// class StreamingAudioPlayer {
// public:
//     StreamingAudioPlayer(int sampleRate = 44100,
//                          size_t bufferSize = 8192)
//             : m_sampleRate(sampleRate),
//               m_buffer(bufferSize),
//               m_isRunning(false),
//               m_playThread(),
//               m_stream(nullptr) {
//         Pa_Initialize();
//     }

//     ~StreamingAudioPlayer() {
//         stop();
//         Pa_Terminate();
//     }

//     void start() {
//         if (!m_isRunning) {
//             m_isRunning = true;
//             m_playThread = std::thread(&StreamingAudioPlayer::playbackThread, this);
//         }
//     }

//     void stop() {
//         if (m_isRunning) {
//             m_isRunning = false;
//             if (m_playThread.joinable()) {
//                 m_buffer.stop();
//                 m_playThread.join();
//             }
//         }
//     }

//     bool writeAudio(const float* data, size_t samples) {
//         return m_buffer.write(data, samples);
//     }

//     size_t available() const {
//         return m_buffer.availableWrite();
//     }

//     int getSampleRate() const {
//         return m_sampleRate;
//     }

// private:
//     RingBuffer<float> m_buffer;
//     int m_sampleRate;
//     std::atomic<bool> m_isRunning;
//     std::thread m_playThread;
//     PaStream* m_stream;

//     void playbackThread() {
//         PaStreamParameters outputParams;
//         outputParams.device = Pa_GetDefaultOutputDevice();
//         outputParams.channelCount = 1;
//         outputParams.sampleFormat = paFloat32;
//         outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
//         outputParams.hostApiSpecificStreamInfo = nullptr;

//         Pa_OpenStream(
//                 &m_stream,
//                 nullptr,
//                 &outputParams,
//                 m_sampleRate,
//                 paFramesPerBufferUnspecified,
//                 paClipOff,
//                 [](const void* input, void* output,
//                    unsigned long frameCount,
//                    const PaStreamCallbackTimeInfo* timeInfo,
//                    PaStreamCallbackFlags statusFlags,
//                    void* userData) -> int {
//                     auto* player = static_cast<StreamingAudioPlayer*>(userData);
//                     return player->audioCallback(output, frameCount);
//                 },
//                 this);

//         Pa_StartStream(m_stream);

//         while (m_isRunning) {
// //            // 检查是否需要退出
// //            if (!m_isRunning) {
// //                break;
// //            }
//             Pa_Sleep(100); // 休眠时间可以适当缩短
//         }

//         Pa_StopStream(m_stream);
//         Pa_CloseStream(m_stream);
//     }

//     int audioCallback(void* outputBuffer, unsigned long framesRequested) {
//         float* out = static_cast<float*>(outputBuffer);
//         const size_t samplesRequested = framesRequested; // 单声道
//         size_t samplesRead = m_buffer.read(out, samplesRequested);

//         // 数据不足时填充静音
//         if (samplesRead < samplesRequested) {
//             std::memset(out + samplesRead, 0, (samplesRequested - samplesRead) * sizeof(float));
//             // 可选：触发下溢事件
//         }

//         return paContinue;
//     }
// };

// #endif //WAV2WAV_AUDIO_PLAY_H
