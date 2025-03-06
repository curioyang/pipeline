
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
#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

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

void mic_proc(std::unique_ptr<VadIterator> &vad, NNCASEModel &whisper, NNCASEModel &adapter, NNCASEModel &lit_gpt,
     std::unique_ptr<tokenizers::Tokenizer> &tokenizer)
{
    unsigned int sample_rate=16000;
    int num_channels=1;
    std::vector<float> wav(512, 0.f);
    std::vector<float> audio;
    vad->reset_states();
    bool triggering = false;
    bool pcm_running = false;
    std::cout << "please enter any string to start, \"/bye\" to exit" << std::endl;
    std::string input;
    std::getline(std::cin, input);
    if (input == "/bye") {
        mic_stop = true;
    }

    while (!mic_stop)
    {
        if (!pcm_running) {
            initPcm(sample_rate, num_channels);
            pcm_running = true;
        }

        {
            ScopedTiming st("getPcm");
            getPcm(wav);
        }

        vad->predict(wav);
        if (!vad->is_triggered() && !triggering)
            continue;


        if (vad->is_triggered())
        {
            audio.insert(audio.end(), wav.begin(), wav.end());
            triggering = true;
            continue;
        }
        else
        {
            audio.insert(audio.end(), wav.begin(), wav.end());
            triggering = false;
            deinitPcm();
            pcm_running = false;
        }

        auto [mel, length] = load_audio(audio);
        auto [audio_feature, input_ids] = generate_input_ids<NNCASEModel>(whisper, mel, length);

        // 执行生成
        auto text = A1_A2<NNCASEModel>(audio_feature, input_ids, length, adapter, lit_gpt, tokenizer);
        std::cout << "Generated text: " << text << std::endl;
        audio.clear();
        std::cout << "please enter any string to start, \"/bye\" to exit" << std::endl;
        std::getline(std::cin, input);
        if (input == "/bye") {
            mic_stop = true;
        }
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
    std::string whisper_model = models_dir + "/whisper/whisper_v2.onnx";
    std::string adapter_model = models_dir + "/adapter/adapter.onnx";
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt_v6.onnx";

    ONNXModel whisper(std::make_unique<RuntimeManager>("whisper"), whisper_model);

    ONNXModel adapter(std::make_unique<RuntimeManager>("adapter"), adapter_model);
    ONNXModel lit_gpt(std::make_unique<RuntimeManager>("lit_gpt"), lit_gpt_model);
#else
    std::string vad_model = models_dir + "/vad/vad.kmodel";
    std::string whisper_model = models_dir + "/whisper/whisper.kmodel";
    std::string adapter_model = models_dir + "/adapter/adapter.kmodel";
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt.kmodel";
    NNCASEModel whisper(whisper_model, "whisper");
    NNCASEModel adapter(adapter_model, "adapter");
    NNCASEModel lit_gpt(lit_gpt_model, "lit_gpt");
#endif

    std::string espeak_ng_data = models_dir + "/espeak-ng-data";
    espeak_ng_InitializePath(espeak_ng_data.c_str());
    espeak_ng_ERROR_CONTEXT context = NULL;
    espeak_ng_STATUS result = espeak_ng_Initialize(&context);
    if (result != ENS_OK) {
        espeak_ng_PrintStatusCodeMessage(result, stderr, context);
        espeak_ng_ClearErrorContext(&context);
        exit(1);
    }

    result = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, NULL);
    // samplerate = espeak_ng_GetSampleRate();
    espeak_SetSynthCallback(SynthCallback);

    std::unique_ptr<VadIterator> vad;
#if defined(ONNX)
    vad.reset(new OnnxVadIterator(vad_model));
#else
    vad.reset(new NncaseVadIterator(vad_model));
#endif

    // 处理音频输入
    if (argc == 2)
    {
        mic_proc(vad, whisper, adapter, lit_gpt, tokenizer);
    } else {
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
        auto text = A1_A2<ONNXModel>(audio_feature, input_ids, length, adapter, lit_gpt, tokenizer);
#else
        auto text = A1_A2<NNCASEModel>(audio_feature, input_ids, length, adapter, lit_gpt, tokenizer);
#endif
        std::cout << "Generated text: " << text << std::endl;
    }

    return 0;
}
