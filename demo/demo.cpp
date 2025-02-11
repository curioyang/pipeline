#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "wav2wav.h"
#include "wav.h"
#include "audio.h"

int main(int argc, const char* argv[])
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " model_dir wav_file" << std::endl;
        return 0;
    }

    std::string models_dir = argv[1];
    std::cout << "models dir is: " << models_dir << std::endl;

    std::string vad_model = models_dir + "/vad/silero_vad.onnx";
    std::string whisper_model = models_dir + "/whisper/whisper.onnx";
    std::string adapter_model = models_dir + "/adapter/adapter.onnx";
    std::string wte_model = models_dir + "/wte/wte.onnx";
    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt.onnx";

    ONNXModel whisper(std::make_unique<RuntimeManager>("whisper"), whisper_model);
    ONNXModel adapter(std::make_unique<RuntimeManager>("adapter"), adapter_model);
    ONNXModel wte(std::make_unique<RuntimeManager>("wte"), wte_model );
    ONNXModel lit_gpt(std::make_unique<RuntimeManager>("lit_gpt"), lit_gpt_model );

    // Tokenizer
    std::string tokenizer_file = models_dir + "/../checkpoint/tokenizer.json";
    auto blob = load_bytes_from_file(tokenizer_file);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);

    // 处理音频输入
#if VAD_ENABLE
    wav::WavReader wav_reader(argv[2]);
    std::vector<float> input_wav(wav_reader.num_samples());
    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }

    std::unique_ptr<VadIterator> vad;
    vad.reset(new OnnxVadIterator(vad_model));
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
#else
    auto [mel, length] = load_audio(argv[2]);
#endif
    auto [audio_feature, input_ids] = generate_input_ids(whisper, mel, length);

    // 执行生成
    auto text = A1_A2(audio_feature, input_ids, length, adapter, wte, lit_gpt, tokenizer);
    std::cout << "Generated text: " << text << std::endl;

    return 0;
}
