#include "wav2wav.h"
#include "audio.h"
#include <fstream>
#include <stdlib.h>
#include <iostream>


int main(int argc, const char* argv[]) {
//    if (argc < 2) {
//        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt>" << std::endl;
//        return 0;
//    }

    std::string models_dir = argv[1];
    std::cout << "models dir is: " << models_dir << std::endl;

    std::string whisper_model = models_dir + "/whisper/whisper.onnx";
//    std::string adapter_model = models_dir + "/adapter/adapter.onnx";
//    std::string wte_model = models_dir + "/wte/wte.onnx";
//    std::string lit_gpt_model = models_dir + "/lit_gpt/lit_gpt.onnx";

    ONNXModel whisper(std::make_unique<RuntimeManager>("whisper"), whisper_model);
//    ONNXModel adapter(std::make_unique<RuntimeManager>("adapter"), adapter_model);
//    ONNXModel wte(std::make_unique<RuntimeManager>("wte"), wte_model );
//    ONNXModel lit_gpt(std::make_unique<RuntimeManager>("lit_gpt"), lit_gpt_model );

    // 处理音频输入
    auto [mel, length] = load_audio("../data/output1.wav");
//    auto input_ids = generate_input_ids(whisper, mel, length);
//
//    // 执行生成
//    auto result = generate_AA(mel, input_ids, adapter, wte, lit_gpt);

    // 解码并输出结果
//    std::string text = decode_text(result.text_tokens);
//    std::cout << "Generated text: " << text << std::endl;

    return 0;
}
