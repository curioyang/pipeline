#include<vector>
#include "wav2wav.h"
#include "audio.h"
#include "utils.h"


tensor_info<float> concat_feat(tensor_info<float> &audio_embs, tensor_info<float> &input_embs) {
    auto audio_embs_shape = audio_embs.shape;
    auto input_embs_shape = input_embs.shape;
    auto audio_embs_data = audio_embs.data;
    auto input_embs_data = input_embs.data;

    /*
     * audio_len = audio_emb.shape[1]
     * for i in range(7):
     *      input_embs[i, 0, 1:audio_len + 1, :] = audio_emb[0, :audio_len].copy()
     * return input_embs
     */

    auto audio_len = audio_embs_shape[1];

    for (int i = 0; i < 7; i++) {
        std::memcpy(input_embs_data.data() + i * input_embs_shape[2] * input_embs_shape[3] + input_embs_shape[3],
                    audio_embs_data.data(),
                    audio_len * audio_embs_shape[2] * sizeof(float));
    }
    return {.data = input_embs_data, .shape = input_embs_shape};
}

int sample(tensor_info<float> &logits, float temperature, int top_k, float top_p) {

    std::vector<float> logits_(logits.data.begin(), logits.data.end());

    // Temperature scaling
    if (top_p < 0.0 || top_p > 1.0)
        throw std::runtime_error("top_p must be between 0 and 1");

    // logits_ shape :  1, y, z
    // here get 0, -1.   means last z.
    std::vector<float> logits_part(logits_.end() - logits.shape[2], logits_.end());

    // Top-K筛选
    // torch.topk
    auto K = std::min(top_k, (int) logits_part.size());
    auto topK_result = topK(logits_part, K);
    std::vector<float> value/*(topK_result.size(),0)*/;
    std::vector<long> index/*(topK_result.size(),0)*/;

    for (auto &i: topK_result) {
        value.push_back(i.first);
        index.push_back(i.second);
    }
    return index[0];
//// 从现在的Python代码来看,下面的函数似乎不起作用
////    logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
//    std::fill(logits_part.begin(), logits_part.end(), -INFINITY);
//    for (int i = 0; i < value.size(); i++)
//        logits_part[index[i]] = value[i];
//
//    if (temperature > 0.0 || top_p > 0.0) {
//        // 温度调整
//        if (temperature > 0.0) {
//            for (auto &i: logits_part)
//                i /= temperature;
//        }
//
////        if (top_p < 1.0) {
////            sample_top_p(logits_part, top_p);
//        auto probs = softmax(logits_part);
//
//        //        multinomial_num_samples_1(probs);

//    }
//
//    if (top_k > 0) {
//        std::vector<std::pair<float, int>> pairs;
//        for (int i = 0; i < logits_part.size(); ++i)
//            pairs.emplace_back(logits_[i], i);
//        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
//                          std::greater<>());
//        std::fill(logits_.begin(), logits_.end(), -INFINITY);
//        for (int i = 0; i < top_k; ++i)
//            logits_[pairs[i].second] = pairs[i].first;
//    }
//
//    // Top-P采样
//    if (top_p < 1.0f) {
//        std::vector<std::pair<float, int>> pairs;
//        for (int i = 0; i < logits_part.size(); ++i)
//            pairs.emplace_back(logits_[i], i);
//        std::sort(pairs.begin(), pairs.end(), std::greater<>());
//
//        float cumulative = 0.0f;
//        int cutoff = 0;
//        while (cumulative < top_p && cutoff < logits_part.size()) {
//            cumulative += expf(pairs[cutoff].first);
//            cutoff++;
//        }
//
//        std::fill(logits_.begin(), logits_.end(), -INFINITY);
//        for (int i = 0; i < cutoff; ++i)
//            logits_[pairs[i].second] = pairs[i].first;
//    }
//
//    // 多项式采样
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::discrete_distribution<> dist(logits_.begin(), logits_.end());
//    return dist(gen);
}

std::tuple<std::vector<long>, int, tensor_info<float>, tensor_info<float>>
next_token_A1T2(ONNXModel &gpt, tensor_info<float> &input_embs_concat, tensor_info<long> &input_pos_tensor,
                tensor_info<float> &past_ks_tensor, tensor_info<float> &past_vs_tensor, int sub_step,
                float temperature, int top_k, float top_p) {
    std::vector<Value> inputs;

    auto input_embs = Input<float>(input_embs_concat.shape, gpt.runtime_manager_);
    auto input_embs_ptr = input_embs.GetTensorMutableData<float>();
    std::memcpy(input_embs_ptr, input_embs_concat.data.data(), input_embs_concat.data.size() * sizeof(float));
    inputs.emplace_back(std::move(input_embs));

    auto past_ks_ = Input<float>(past_ks_tensor.shape, gpt.runtime_manager_);
    auto past_ks_ptr = past_ks_.GetTensorMutableData<float>();
    std::memcpy(past_ks_ptr, past_ks_tensor.data.data(), past_ks_tensor.data.size() * sizeof(float));
    inputs.emplace_back(std::move(past_ks_));

    auto past_vs_ = Input<float>(past_vs_tensor.shape, gpt.runtime_manager_);
    auto past_vs_ptr = past_vs_.GetTensorMutableData<float>();
    std::memcpy(past_vs_ptr, past_vs_tensor.data.data(), past_vs_tensor.data.size() * sizeof(float));
    inputs.emplace_back(std::move(past_vs_));

    auto input_pos = Input<long>(input_pos_tensor.shape, gpt.runtime_manager_);
    auto input_pos_ptr = input_pos.GetTensorMutableData<long>();
    std::memcpy(input_pos_ptr, input_pos_tensor.data.data(), input_pos_tensor.data.size() * sizeof(long));
    inputs.emplace_back(std::move(input_pos));

    auto gpt_output = gpt.onForward(inputs);

    auto logits_a = gpt.get_result_vector<float>(gpt_output, 0);
    auto logit_t = gpt.get_result_vector<float>(gpt_output, 1);
    auto next_ks = gpt.get_result_vector<float>(gpt_output, 2);
    auto next_vs = gpt.get_result_vector<float>(gpt_output, 3);

    std::vector<long> next_audio_tokens;
    for (int i = 0; i < logits_a.shape[0]; i++) {
        std::vector<float> logits_a_i_data(
                std::vector<float>(logits_a.data.data() + i * logits_a.shape[2] * logits_a.shape[3],
                                   logits_a.data.data() + (i + 1) * logits_a.shape[2] * logits_a.shape[3]));
        tensor_info<float> logits_a_i{.data = logits_a_i_data, .shape = {1, logits_a.shape[2], logits_a.shape[3]}};
        auto next_a = sample(logits_a_i, temperature, top_k, top_p);

        next_audio_tokens.emplace_back(next_a);
    }
    auto next_t = sample(logit_t, temperature, top_k, top_p);
    return std::tuple(next_audio_tokens, next_t, next_ks, next_vs);
}


std::vector<std::vector<long>>
generate_AA(std::vector<std::vector<float>> &audio_feature, std::vector<std::vector<long>> &input_ids,
            ONNXModel &adapter, ONNXModel &wte, ONNXModel &gpt,
            int max_returned_tokens = 2048,
            float temperature = 0.9,
            int top_k = 1,
            float top_p = 1,
            int eos_id_a = _eoa,
            int eos_id_t = _eot,
            int pad_id_t = _pad_t,
            int shift = padded_text_vocabsize,
            bool include_prompt = true,
            bool generate_text = false) {
    auto T = input_ids[0].size();
    std::vector<std::vector<long>> outputs(8);

    // adapter
    std::vector<long> audio_shape = {1, (int) audio_feature.size(), (int) audio_feature[0].size()};
    auto adapter_input = Input<float>(audio_shape, adapter.runtime_manager_);
    auto adapter_ptr = adapter_input.GetTensorMutableData<float>();
    for (int i = 0; i < audio_feature.size(); ++i) {
        for (int j = 0; j < audio_feature[0].size(); ++j) {
            *adapter_ptr++ = audio_feature[i][j];
        }
    }

    std::vector<Value> adapter_inputs;
    adapter_inputs.emplace_back(std::move(adapter_input));

    auto adapter_output = adapter.onForward(adapter_inputs);
    auto audio_embs = adapter.get_result_vector<float>(adapter_output, 0);

    std::vector<long> input_ids_shape = {(int) input_ids.size(), 1, (int) input_ids[0].size()};
    auto wte_input = Input<long>(input_ids_shape, wte.runtime_manager_);
    auto wte_ptr = wte_input.GetTensorMutableData<long>();
    for (int i = 0; i < input_ids.size(); ++i) {
        for (int j = 0; j < input_ids[0].size(); ++j) {
            *wte_ptr++ = input_ids[i][j];
        }
    }

    std::vector<Value> wte_inputs;
    wte_inputs.emplace_back(std::move(wte_input));
    auto wte_output = wte.onForward(wte_inputs);
    auto input_embs = wte.get_result_vector<float>(wte_output, 0);


    auto input_embs_concat = concat_feat(audio_embs, input_embs);

    std::vector<float> past_ks(0, 0);
    std::vector<float> past_vs(0, 0);
    std::vector<long> input_pos;
    for (int i = 0; i < T; i++)
        input_pos.emplace_back(i);

    tensor_info<float> past_ks_tensor{.data=past_ks, .shape={24, 1, 14, 0, 64}};
    tensor_info<float> past_vs_tensor{.data=past_ks, .shape={24, 1, 14, 0, 64}};
    tensor_info<long> input_pos_tensor{.data=input_pos, .shape={(long) input_pos.size()}};


    auto [tokens_A, token_T, past_ks_, past_vs_] = next_token_A1T2(gpt, input_embs_concat, input_pos_tensor,
                                                                   past_ks_tensor, past_vs_tensor, 1, temperature,
                                                                   top_k,
                                                                   top_p);
//    past_ks_tensor = past_ks_;
//    past_vs_tensor = past_vs_;

    for (int i = 0; i < 7; i++)
        outputs[i].emplace_back(tokens_A[i]);
    outputs[7].emplace_back(token_T);
    input_pos.resize(1);
    input_pos[0] = (long) T;

    bool text_end = false;
    for (int sub_step = 2; sub_step < max_returned_tokens - T + 1; sub_step++) {
        std::vector<long> model_input_ids;
        for (int i = 0; i < 7; i++) {
            model_input_ids.emplace_back((long) tokens_A[i] + 152000 + i * 4160);
        }
        model_input_ids.emplace_back((long) token_T);
        tensor_info<long> input_ids_tensor{.data=model_input_ids, .shape={(long) model_input_ids.size(), 1, 1}};
        std::vector<Value> wte_inputs_loop;
        auto wte_input_ = Input<long>(input_ids_tensor.shape, wte.runtime_manager_);
        auto wte_input_ptr = wte_input_.GetTensorMutableData<long>();
        std::memcpy(wte_input_ptr, input_ids_tensor.data.data(), input_ids_tensor.data.size() * sizeof(long));
        wte_inputs_loop.emplace_back(std::move(wte_input_));

        auto wte_output_loop = wte.onForward(wte_inputs_loop);
        auto input_embs_loop_tensor = wte.get_result_vector<float>(wte_output_loop, 0);

        tensor_info<long> input_pos_loop_tensor{.data=input_pos, .shape={(long) input_pos.size()}};
        std::tuple<std::vector<long>, int, tensor_info<float>, tensor_info<float>>
                (tokens_A, token_T, past_ks_, past_vs_) = next_token_A1T2(gpt,
                                                                          input_embs_loop_tensor,
                                                                          input_pos_loop_tensor,
                                                                          past_ks_,
                                                                          past_vs_,
                                                                          sub_step,
                                                                          temperature,
                                                                          top_k,
                                                                          top_p);
//        past_ks_tensor = past_ks_;
//        past_vs_tensor = past_vs_;
//        tokens_A = tokens_A_loop;
//        token_T = token_T_loop;
        if (text_end)
            token_T = pad_id_t;
        if ((int) tokens_A[-1] == eos_id_a)
            break;
        if (token_T == eos_id_t)
            text_end = true;

        for (int i = 0; i < 7; i++) {
            outputs[i].emplace_back(tokens_A[i]);
        }
        outputs[7].emplace_back(token_T);
        input_pos[0] += 1;
    }

    return outputs;
}

/*GenerationResult*/void A1_A2(std::vector<std::vector<float>> &audio_feature,
                               std::vector<std::vector<int64_t>> &input_ids,
                               int length,
                               ONNXModel &adapter,
                               ONNXModel &wte,
                               ONNXModel &gpt) {
    auto tokenizer_list = generate_AA(audio_feature, input_ids, adapter, wte, gpt,
                                      2048,
                                      0.9,
                                      1,
                                      1.0,
                                      _eoa,
                                      _eot,
                                      _pad_t,
                                      padded_text_vocabsize,
                                      true,
                                      false);
    for (auto &i: tokenizer_list) {
        for (auto &j: i)
            std::cout << j << " ";
        std::cout << std::endl;
    }
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


std::pair<std::vector<std::vector<float>>, std::vector<std::vector<long>>>
generate_input_ids(ONNXModel &model, std::vector<std::vector<float>> &mel, int length,
                   int step,
                   int special_token_a, int special_token_t) {
    std::vector<long> mel_shape = {1, (int) mel.size(), (int) mel[0].size()};
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

    std::vector<std::vector<float>> audio_feature(length, std::vector<float>(output_shape[2], 0));
    std::cout << "audio_feature shape: " << audio_feature.size() << " " << audio_feature[0].size() << std::endl;
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < output_shape[2]; ++j) {
            audio_feature[i][j] = output_data[i * output_shape[2] + j];
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


