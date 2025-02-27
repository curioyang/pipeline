
#include "audio.h"
#include "wav.h"
#include "wav2wav.h"
#include <csignal>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

#ifdef ONNX
#include "ONNXWrapper.h"
using namespace omni_onnx;
#else
#include "NNCASEWrapper.h"
#if __riscv
#include "get_pcm.h"
#include "play_pcm.h"
#endif

using namespace nncase::runtime;
void __attribute__((destructor)) cleanup()
{
    std::cout << "Cleaning up memory..." << std::endl;
    shrink_memory_pool();
}
#endif

void signal_handler(int signum)
{
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}

#if __riscv
std::atomic<bool> mic_stop(false);

void mic_proc(std::unique_ptr<VadIterator> &vad, NNCASEModel &whisper, NNCASEModel &adapter, NNCASEModel &lit_gpt, NNCASEModel &snac, \
     std::unique_ptr<tokenizers::Tokenizer> &tokenizer, StreamingAudioPlayer &player)
{
    unsigned int sample_rate=16000;
    int num_channels=1;
    std::vector<float> wav;
    std::vector<float> audio;
    vad->reset_states();
    bool triggering = false;
    bool pcm_running = false;
    std::cout << "please enter any string to start, \"bye\" to exit" << std::endl;
    std::string input;
    std::cin >> input;
    while (!mic_stop)
    {
        if (!pcm_running) {
            std::cout << "initPcm" << std::endl;
            initPcm(sample_rate, num_channels);
            pcm_running = true;
        }

        wav.clear();
        getPcm(wav);
        // process wav with fp32
        for (size_t i = 0; i < wav.size(); i++)
            wav[i] = wav[i] / 32768;

        vad->predict(wav);
        if (!vad->is_triggered() && !triggering)
            continue;


        if (vad->is_triggered())
        {
            audio.insert(audio.end(), wav.begin(), wav.end());
            triggering = true;
            std::cout << "vad is triggering" << std::endl;
            continue;
        }
        else
        {
            std::cout << "vad is not triggering: triggering = " << triggering << std::endl;
            audio.insert(audio.end(), wav.begin(), wav.end());
            triggering = false;
            deinitPcm();
            pcm_running = false;
        }

        auto [mel, length] = load_audio(audio);
        auto [audio_feature, input_ids] = generate_input_ids<NNCASEModel>(whisper, mel, length);

        // 执行生成
        auto text = A1_A2<NNCASEModel>(audio_feature, input_ids, length, adapter, lit_gpt, snac, tokenizer, player);
        std::cout << "Generated text: " << text << std::endl;
        audio.clear();
        std::cout << "please enter any string to start, \"bye\" to exit" << std::endl;
        std::cin >> input;
    }
}
#endif


int main(int argc, const char* argv[])
{
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    // 设置多个信号处理器
    sigaction(SIGINT, &sa, nullptr);  // Ctrl+C
    sigaction(SIGTERM, &sa, nullptr); // 终止信号
    sigaction(SIGSEGV, &sa, nullptr); // 段错误
    sigaction(SIGALRM, &sa, nullptr); // 定时器信号
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir [wav_file]" << std::endl;
        return 0;
    }

    std::string models_dir = argv[1];
    std::cout << "models dir is: " << models_dir << std::endl;

    // Tokenizer
    std::string tokenizer_file = models_dir + "/../checkpoint/tokenizer.json";
    auto blob = load_bytes_from_file(tokenizer_file);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);

#if defined(ONNX)
    std::string vad_model = models_dir + "/vad/silero_vad.onnx";
    // whisper.onnx is 3000 whisper_v2.onnx is 1500
    // if change model, modify common.h:37L
    std::string whisper_model = models_dir + "/whisper/whisper.onnx";
    std::string adapter_model = models_dir + "/adapter/adapter.onnx";
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt_v6.onnx";
    std::string snac_model = models_dir + "/snac/snac.onnx";

    ONNXModel whisper(std::make_unique<RuntimeManager>("whisper"), whisper_model);

    ONNXModel adapter(std::make_unique<RuntimeManager>("adapter"), adapter_model);
    ONNXModel lit_gpt(std::make_unique<RuntimeManager>("lit_gpt"), lit_gpt_model);
    ONNXModel snac(std::make_unique<RuntimeManager>("snac"), snac_model);
#else
    // nncase::runtime::shrink_memory_pool();
    std::string vad_model = models_dir + "/vad/vad.kmodel";
    std::string whisper_model = models_dir + "/whisper/whisper.kmodel";
    std::string adapter_model = models_dir + "/adapter/adapter.kmodel";
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt.kmodel";
    // TODO: change snac to audio_0 length is 8
    std::string snac_model = models_dir + "/snac/snac.kmodel";
    NNCASEModel whisper(whisper_model, "whisper");
    NNCASEModel adapter(adapter_model, "adapter");
    NNCASEModel lit_gpt(lit_gpt_model, "lit_gpt");
    NNCASEModel snac(snac_model,"snac");
#endif

    std::unique_ptr<VadIterator> vad;
#if defined(ONNX)
    vad.reset(new OnnxVadIterator(vad_model));
#else
    vad.reset(new NncaseVadIterator(vad_model));
#endif

    // init audio Player
    int buffer_size = 960 * 1024;
    StreamingAudioPlayer audioplayer(24000, buffer_size); // 24kHz, 4KB buffer
    audioplayer.start();

    // 处理音频输入
#if __riscv
// #if 0
    // std::thread thread_mic(mic_proc, vad, whisper, adapter, lit_gpt, snac, tokenizer);
    // while (getchar() != 'q')
    // {
    //     usleep(10000);
    // }
    // mic_stop = true;
    // thread_mic.join();
    // initPlayer(24000, 1, 960, 16);
    mic_proc(vad, whisper, adapter, lit_gpt, snac, tokenizer, audioplayer);
    // deinitPlayer();
#else
    wav::WavReader wav_reader(argv[2]);
    std::vector<float> input_wav(wav_reader.num_samples());
    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }

    vad->process(input_wav);

    // get_speech_timestamps
    auto stamps = vad->get_speech_timestamps();
    assert(!stamps.empty());
    for (int i = 0; i < stamps.size(); i++)
    {
        std::cout << stamps[i].c_str() << std::endl;
    }

    std::vector<float> audio(input_wav.begin() + stamps.front().start, input_wav.begin() + stamps.back().end);
    auto [mel, length] = load_audio(audio);

#if defined(ONNX)
    auto [audio_feature, input_ids] = generate_input_ids<ONNXModel>(whisper, mel, length);
#else
    auto [audio_feature, input_ids] = generate_input_ids<NNCASEModel>(whisper, mel, length);
#endif

    // 执行生成
#if defined(ONNX)
    auto text = A1_A2<ONNXModel>(audio_feature, input_ids, length, adapter, lit_gpt, snac, tokenizer, audioplayer);
#else
    auto text = A1_A2<NNCASEModel>(audio_feature, input_ids, length, adapter, lit_gpt, snac, tokenizer, audioplayer);
#endif
    std::cout << "Generated text: " << text << std::endl;
#endif

    while (audioplayer.available() < buffer_size) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    audioplayer.stop();
    return 0;
}
