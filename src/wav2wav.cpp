#include<vector>
#include "wav2wav.h"
#include "audio.h"

GenerationResult generate_AA(const std::vector<float>& mel,
                             const std::vector<std::vector<int64_t>>& input_ids,
                             ONNXModel& adapter,
                             ONNXModel& wte,
                             ONNXModel& gpt)
{
    //    // 初始化模型输入
    //    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    //        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    //
    //    // 运行适配器模型
    //    std::vector<Ort::Value> adapter_inputs;
    //    adapter_inputs.emplace_back(Ort::Value::CreateTensor<float>(
    //        memory_info, const_cast<float *>(mel.data()), mel.size(), {1, 80, 3000}));
    //    auto adapter_out = adapter.run({"mel"}, adapter_inputs, {"audio_embs"});
    //
    //    // 生成嵌入
    //    std::vector<Ort::Value> wte_inputs;
    //    // 构建输入ID张量...
    //    auto wte_out = wte.run({"input_ids"}, wte_inputs, {"embeddings"});
    //
    //    // 主生成循环
    //    std::vector<int> generated_tokens;
    //    std::vector<float> past_keys, past_values;
    //
    //    while (!stop_condition)
    //    {
    //        // 构建GPT输入
    //        Ort::Value inputs[] = {embeddings_tensor, keys_tensor, values_tensor};
    //        auto gpt_out = gpt.run({"input_embs", "past_keys", "past_values"},
    //                               {inputs, 3}, {"logits", "new_keys", "new_values"});
    //
    //        // 采样并更新状态
    //        int next_token = sample(gpt_out[0].GetTensorData<float>(),
    //                                gpt_out[0].GetTensorTypeAndShapeInfo().GetElementCount(),
    //                                0.9f, 1, 1.0f);
    //        generated_tokens.push_back(next_token);
    //
    //        // 更新past keys/values
    //        past_keys = process_new_keys(gpt_out[1]);
    //        past_values = process_new_values(gpt_out[2]);
    //    }
    //
    //    return {audio_tokens, text_tokens};
}


std::vector<std::vector<int64_t>> generate_input_ids(ONNXModel& model, std::vector<std::vector<float>>& mel, int length,
                                                     int step,
                                                     int special_token_a, int special_token_t)
{
    std::vector<size_t> mel_shape = {1, mel.size(), mel[0].size()};

    model.onForward({{"mel", mel_shape, mel}});

    std::vector<std::vector<int64_t>> input_ids(8);
    for (int i = 0; i < 7; ++i)
    {
        input_ids[i].push_back(_input_a + i * padded_audio_vocab);
        input_ids[i].insert(input_ids[i].end(), length, _pad_a + i * padded_audio_vocab);
        input_ids[i].push_back(_eoa + i * padded_audio_vocab);
        input_ids[i].push_back(_answer_a + i * padded_audio_vocab);
    }
    // input_ids[7] = std::vector<int64_t>{_input_t, std::vector<int64_t>(length, _pad_t), _eot, _answer_t};
    return input_ids;
}

int sample(const float* logits, int size, float temp, int top_k, float top_p)
{
    std::vector<float> probs(logits, logits + size);

    // Temperature scaling
    for (auto& p : probs) p /= temp;

    // Top-K筛选
    if (top_k > 0)
    {
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
    if (top_p < 1.0f)
    {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < size; ++i)
            pairs.emplace_back(probs[i], i);
        std::sort(pairs.begin(), pairs.end(), std::greater<>());

        float cumulative = 0.0f;
        int cutoff = 0;
        while (cumulative < top_p && cutoff < size)
        {
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
