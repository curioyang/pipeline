#include<vector>
#include "wav2wav.h"
#include "audio.h"
#include "utils.h"

void generate_AA(std::vector<std::vector<float>> audio_feature, std::vector<std::vector<long>> input_ids, ONNXModel& adapter, ONNXModel& wte, ONNXModel& gpt,
                 int max_returned_tokens=2048,
                 float temperature=0.9,
                 int top_k=1,
                 int eos_id_a=_eoa,
                 int eos_id_t=_eot,
                 int pad_id_t=_pad_t,
                 int shift=padded_text_vocabsize,
                 bool include_prompt=true,
                 bool generate_text=true)
{

}

/*GenerationResult*/void A1_A2(std::vector<std::vector<float>> &audio_feature,
                       std::vector<std::vector<int64_t>> &input_ids,
                       int length,
                       ONNXModel &adapter,
                       ONNXModel &wte,
                       ONNXModel &gpt) {
    generate_AA(audio_feature, input_ids, adapter, wte, gpt,
                2048,
                0.9,
                1,
                _eoa,
                _eot,
                _pad_t,
                padded_text_vocabsize,
                true,
                true);

//    return {tokenlist, tokenlist};
//    tokenlist = tokenlist[-1]
//    if text_vocabsize in tokenlist:
//    tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
//    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()
}

//{
//// 初始化模型输入
//Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
//        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
//
//// 运行适配器模型
//std::vector<Ort::Value> adapter_inputs;
//adapter_inputs.emplace_back(Ort::Value::CreateTensor<float>(
//        memory_info, const_cast<float *>(mel.data()), mel.size(), {1, 80, 3000}));
//auto adapter_out = adapter.run({"mel"}, adapter_inputs, {"audio_embs"});
//
//// 生成嵌入
//std::vector<Ort::Value> wte_inputs;
//// 构建输入ID张量...
//auto wte_out = wte.run({"input_ids"}, wte_inputs, {"embeddings"});
//
//// 主生成循环
//std::vector<int> generated_tokens;
//std::vector<float> past_keys, past_values;
//
//while (!stop_condition)
//{
//// 构建GPT输入
//Ort::Value inputs[] = {embeddings_tensor, keys_tensor, values_tensor};
//auto gpt_out = gpt.run({"input_embs", "past_keys", "past_values"},
//                       {inputs, 3}, {"logits", "new_keys", "new_values"});
//
//// 采样并更新状态
//int next_token = sample(gpt_out[0].GetTensorData<float>(),
//                        gpt_out[0].GetTensorTypeAndShapeInfo().GetElementCount(),
//                        0.9f, 1, 1.0f);
//generated_tokens.push_back(next_token);
//
//// 更新past keys/values
//past_keys = process_new_keys(gpt_out[1]);
//past_values = process_new_values(gpt_out[2]);
//}
//
//return {audio_tokens, text_tokens};
//}


std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>>
generate_input_ids(ONNXModel &model, std::vector<std::vector<float>> &mel, int length,
                   int step,
                   int special_token_a, int special_token_t) {
    std::vector<int> mel_shape = {1, (int) mel.size(), (int) mel[0].size()};
    auto mel_input = Input<float>(mel_shape, model.runtime_manager_);
    std::vector<Value> inputs;
    auto ptr = mel_input.GetTensorMutableData<float>();
    for (int i = 0; i < mel.size(); ++i) {
        for (int j = 0; j < mel[0].size(); ++j) {
            *ptr++ = mel[i][j];
        }
    }

    inputs.emplace_back(std::move(mel_input));
    auto output = model.onForward(inputs);
    auto [output_data, output_shape] = model.get_result_vector<float>(output, 0);

    std::vector<std::vector<float>> audio_feature(length, std::vector<float>(output_shape[-1], 0));
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < output_shape[-1]; ++j) {
            audio_feature[i][j] = output_data[i * output_shape[-1] + j];
        }
    }

    std::vector<std::vector<int64_t>> input_ids(8);
    for (int i = 0; i < 7; ++i) {
        input_ids[i].push_back(_input_a + 152000 + i * 4160);
        input_ids[i].insert(input_ids[i].end(), length, _pad_a + 152000 + i * 4160);
        input_ids[i].push_back(_eoa + +152000 + i * 4160);
        input_ids[i].push_back(special_token_a + +152000 + i * 4160);

    }
//    input_ids[7] = std::vector<int64_t>{_input_t, std::vector<int64_t>(length, _pad_t), _eot, _answer_t};
    input_ids[7].push_back(_input_t);
    input_ids[7].insert(input_ids[7].end(), length, _pad_t);
    input_ids[7].push_back(_eot);
    input_ids[7].push_back(special_token_t);

    return {audio_feature, input_ids};
}

int sample(const float *logits, int size, float temp, int top_k, float top_p) {
    std::vector<float> probs(logits, logits + size);

    // Temperature scaling
    for (auto &p: probs) p /= temp;

    // Top-K筛选
    if (top_k > 0) {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < size; ++i)
            pairs.emplace_back(probs[i], i);
        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                          std::greater<>());
        std::fill(probs.begin(), probs.end(), -INFINITY);
        for (int i = 0; i < top_k; ++i)
            probs[pairs[i].second] = pairs[i].first;
    }

    // Top-P采样
    if (top_p < 1.0f) {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < size; ++i)
            pairs.emplace_back(probs[i], i);
        std::sort(pairs.begin(), pairs.end(), std::greater<>());

        float cumulative = 0.0f;
        int cutoff = 0;
        while (cumulative < top_p && cutoff < size) {
            cumulative += expf(pairs[cutoff].first);
            cutoff++;
        }

        std::fill(probs.begin(), probs.end(), -INFINITY);
        for (int i = 0; i < cutoff; ++i)
            probs[pairs[i].second] = pairs[i].first;
    }

    // 多项式采样
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}
