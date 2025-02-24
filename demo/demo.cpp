
#include "audio.h"
#include "wav.h"
#include "wav2wav.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

#ifdef BUILD_ONNX
#include "ONNXWrapper.h"
using namespace omni_onnx;
#else
#include "NNCASEWrapper.h"
#endif

#define whisper_ONNX 
#define adapter_ONNX
#define lit_gpt_ONNX
// #define snac_ONNX

#ifdef whisper_ONNX
#define whisper_TYPE ONNXModel
#else
#define whisper_TYPE NNCASEModel
#endif

#ifdef adapter_ONNX
#define adapter_TYPE ONNXModel
#else
#define adapter_TYPE NNCASEModel
#endif

#ifdef lit_gpt_ONNX
#define lit_gpt_TYPE ONNXModel
#else
#define lit_gpt_TYPE NNCASEModel
#endif

#ifdef snac_ONNX
#define snac_TYPE ONNXModel
#else
#define snac_TYPE NNCASEModel
#endif

int main(int argc, const char* argv[])
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " model_dir wav_file" << std::endl;
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
#else
    std::string vad_model = models_dir + "/vad/vad.kmodel";
#endif

#ifdef whisper_ONNX
    std::string whisper_model = models_dir + "/whisper/whisper.onnx";
    ONNXModel whisper(std::make_unique<RuntimeManager>("whisper"), whisper_model);
#else
    std::string whisper_model = models_dir + "/whisper/kmodel/test.kmodel";
    NNCASEModel whisper(whisper_model, "whisper");
#endif

#ifdef adapter_ONNX
    std::string adapter_model = models_dir + "/adapter/adapter.onnx";
    ONNXModel adapter(std::make_unique<RuntimeManager>("adapter"), adapter_model);
#else
    std::string adapter_model = models_dir + "/adapter/adapter.kmodel";
    NNCASEModel adapter(adapter_model, "adapter");
#endif

#ifdef lit_gpt_ONNX
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt_v4.onnx";
    ONNXModel lit_gpt(std::make_unique<RuntimeManager>("lit_gpt"), lit_gpt_model);
#else
    std::string lit_gpt_model = models_dir + "/lit_gpt/kmodel/test_128.kmodel";
    NNCASEModel lit_gpt(lit_gpt_model, "lit_gpt");
#endif

#ifdef snac_ONNX
    // std::string snac_model = models_dir + "/snac/snac_8_16_32.onnx";
    std::string snac_model = models_dir + "/snac/snac.onnx";
    ONNXModel snac(std::make_unique<RuntimeManager>("snac"), snac_model);
#else
    // std::string snac_model = models_dir + "/snac/kmodel/test.kmodel";
    std::string snac_model = models_dir + "/snac/kmodel/test_8.kmodel";
    NNCASEModel snac(snac_model,"snac");
#endif



    // 处理音频输入
#if VAD_ENABLE
    wav::WavReader wav_reader(argv[2]);
    std::vector<float> input_wav(wav_reader.num_samples());
    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }

    std::unique_ptr<VadIterator> vad;
#if defined(ONNX)
    vad.reset(new OnnxVadIterator(vad_model));
#else
    vad.reset(new NncaseVadIterator(vad_model));
#endif
    vad->process(input_wav);

    // get_speech_timestamps
    auto stamps = vad->get_speech_timestamps();
    assert(!stamps.empty());
    for (int i = 0; i < stamps.size(); i++)
    {
        std::cout << stamps[i].c_str() << std::endl;
    }
    // vad = nullptr;

    std::vector<float> audio(input_wav.begin() + stamps.front().start, input_wav.begin() + stamps.back().end);
    auto [mel, length] = load_audio(audio);
#else
    auto [mel, length] = load_audio(argv[2]);
#endif

// #if defined(ONNX)
//     auto [audio_feature, input_ids] = generate_input_ids<ONNXModel>(whisper, mel, length);
// #else
    auto [audio_feature, input_ids] = generate_input_ids<whisper_TYPE>(whisper, mel, length);
// #endif
    // // init audio Player
    // int buffer_size = 4096;
    // StreamingAudioPlayer audioplayer(24000, buffer_size); // 24kHz, 4KB buffer
    // audioplayer.start();

    // 执行生成
// #if defined(ONNX)
//     auto text = A1_A2<ONNXModel>(audio_feature, input_ids, length, adapter, lit_gpt, snac, tokenizer);
// #else
    auto text = A1_A2<adapter_TYPE, lit_gpt_TYPE, snac_TYPE>(audio_feature, input_ids, length, adapter, lit_gpt, snac, tokenizer);
// #endif
    std::cout << "Generated text: " << text << std::endl;
    // while (audioplayer.available() < buffer_size) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }

    // audioplayer.stop();
    return 0;
}
