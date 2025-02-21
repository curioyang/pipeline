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
}

std::string load_bytes_from_file(const std::string &path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

#if defined(ONNX)
template<class T1, class T2>
tensor_info<T2> model_run(ONNXModel &model, tensor_info<T1> &input_ids) {
    auto input = Input<T1>(input_ids, model.runtime_manager_);
    std::vector<Value> model_inputs;
    model_inputs.emplace_back(std::move(input));
    auto output = model.onForward(model_inputs);
    auto result = model.get_result_vector<T2>(output, 0);

    return std::move(result);
}

template<class T1, class T2>
tensor_info<T2> model_run(ONNXModel &model, std::vector<tensor_info<T1>> &input_ids) {
    std::vector<Value> model_inputs;
    for(auto &it: input_ids)
    {
        auto input = Input<T1>(it, model.runtime_manager_);
        model_inputs.emplace_back(std::move(input));
    }
    auto output = model.onForward(model_inputs);
    auto result = model.get_result_vector<T2>(output, 0);

    return std::move(result);
}

std::tuple<std::vector<int>, int, tensor_info<float>, tensor_info<float>>
next_token_A1T2(ONNXModel &gpt, tensor_info<float> &input_embs_concat, tensor_info<long> &input_pos_tensor,
                tensor_info<float> &past_ks_tensor, tensor_info<float> &past_vs_tensor, int sub_step,
                float temperature, int top_k, float top_p) {
    std::vector<Value> inputs;

    auto input_embs = Input(input_embs_concat, gpt.runtime_manager_);
    auto input_embs_ptr = input_embs.GetTensorMutableData<float>();
    std::memcpy(input_embs_ptr, input_embs_concat.data.data(), input_embs_concat.data.size() * sizeof(float));
    inputs.emplace_back(std::move(input_embs));

    auto past_ks_ = Input(past_ks_tensor, gpt.runtime_manager_);
    auto past_ks_ptr = past_ks_.GetTensorMutableData<float>();
    std::memcpy(past_ks_ptr, past_ks_tensor.data.data(), past_ks_tensor.data.size() * sizeof(float));
    inputs.emplace_back(std::move(past_ks_));

    auto past_vs_ = Input<float>(past_vs_tensor, gpt.runtime_manager_);
    auto past_vs_ptr = past_vs_.GetTensorMutableData<float>();
    std::memcpy(past_vs_ptr, past_vs_tensor.data.data(), past_vs_tensor.data.size() * sizeof(float));
    inputs.emplace_back(std::move(past_vs_));

    auto input_pos = Input<long>(input_pos_tensor, gpt.runtime_manager_);
    auto input_pos_ptr = input_pos.GetTensorMutableData<long>();
    std::memcpy(input_pos_ptr, input_pos_tensor.data.data(), input_pos_tensor.data.size() * sizeof(long));
    inputs.emplace_back(std::move(input_pos));

    auto gpt_output = gpt.onForward(inputs);

    auto logits_a = gpt.get_result_vector<float>(gpt_output, 0);
    auto logit_t = gpt.get_result_vector<float>(gpt_output, 1);
    auto next_ks = gpt.get_result_vector<float>(gpt_output, 2);
    auto next_vs = gpt.get_result_vector<float>(gpt_output, 3);

    std::vector<int> next_audio_tokens;
    for (int i = 0; i < logits_a.shape[0]; i++) {
        std::vector<float> logits_a_i_data(
                std::vector<float>(logits_a.data.data() + i * logits_a.shape[2] * logits_a.shape[3],
                                   logits_a.data.data() + (i + 1) * logits_a.shape[2] * logits_a.shape[3]));
        tensor_info<float> logits_a_i{.data = logits_a_i_data, .shape = {1, logits_a.shape[2], logits_a.shape[3]}};
        auto next_a = sample(logits_a_i, temperature, top_k, top_p);

        next_audio_tokens.emplace_back(next_a);
    }
    auto next_t = sample(logit_t, temperature, top_k, top_p);
    return {next_audio_tokens, next_t, next_ks, next_vs};
}


std::vector<std::vector<int>>
generate_AA(tensor_info<float> &audio_feature, tensor_info<long> &input_ids,
            ONNXModel &adapter, ONNXModel &gpt,
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
    auto T = input_ids.shape[1];
    std::vector<std::vector<int>> outputs(8);

    // adapter
    audio_feature.shape = {1, audio_feature.shape[0], audio_feature.shape[1]};
    auto audio_embs = model_run<float, float>(adapter, audio_feature);

    input_ids.shape = {input_ids.shape[0], 1, input_ids.shape[1]};
    auto input_embs = wte_get_data(input_ids);

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
//         tensor_info<long> input_ids_tensor{.data=model_input_ids, .shape={(long) model_input_ids.size(), 1, 1}};
//         std::vector<Value> wte_inputs_loop;
//         auto wte_input_ = Input<long>(input_ids_tensor.shape, wte.runtime_manager_);
//         auto wte_input_ptr = wte_input_.GetTensorMutableData<long>();
//         std::memcpy(wte_input_ptr, input_ids_tensor.data.data(), input_ids_tensor.data.size() * sizeof(long));
//         wte_inputs_loop.emplace_back(std::move(wte_input_));
//         auto wte_output_loop = wte.onForward(wte_inputs_loop);
//         auto input_embs_loop_tensor = wte.get_result_vector<float>(wte_output_loop, 0);
        tensor_info<long> input_ids_tensor{.data = model_input_ids, .shape={(long) model_input_ids.size(), 1, 1}};
//        auto input_embs_loop_tensor = model_run<long, float>(wte, input_ids_tensor);
        auto input_embs_loop_tensor = wte_get_data(input_ids_tensor);


        tensor_info<long> input_pos_loop_tensor{.data=input_pos, .shape={(long) input_pos.size()}};
//        std::tuple<std::vector<int>, int, tensor_info<float>, tensor_info<float>>
        auto [_tokens_A, _token_T, _past_ks_, _past_vs_] = next_token_A1T2(gpt,
                                                                           input_embs_loop_tensor,
                                                                           input_pos_loop_tensor,
                                                                           past_ks_,
                                                                           past_vs_,
                                                                           sub_step,
                                                                           temperature,
                                                                           top_k,
                                                                           top_p);
        tokens_A = _tokens_A;
        token_T = _token_T;
        past_ks_ = _past_ks_;
        past_vs_ = _past_vs_;

        if (text_end)
            token_T = pad_id_t;
        if ((int) tokens_A[tokens_A.size() - 1] == eos_id_a)
            break;
        if (token_T == eos_id_t)
            text_end = true;

        for (int i = 0; i < 7; i++) {
            outputs[i].emplace_back(tokens_A[i]);
        }
        outputs[7].emplace_back(token_T);
        input_pos[0] += 1;

        // for (int i = 0; i < 8; i++)
        // {
        //     std::cout << "output: [" << i << "] ";
        //     for (auto t : outputs[i])
        //         std::cout << t << " ";
        //     std::cout<<std::endl;
        // }
    }

    return outputs;
}

int countElementsBetweenHashes(const std::vector<int> &lst) {
    try {
        // Find the index of the first '#'
        auto first_index = std::find(lst.begin(), lst.end(), hash_flag);

        if (first_index == lst.end()) {
            throw std::invalid_argument("No '#' found in the list.");
        }

        // Find the index of the second '#' after the first
        auto second_index = std::find(first_index + 1, lst.end(), hash_flag);

        if (second_index == lst.end()) {
            throw std::invalid_argument("Only one '#' found in the list.");
        }

        // Calculate the number of elements between the two indices
        return std::distance(first_index, second_index) - 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1; // 返回-1表示发生错误
    }
}

std::vector<std::vector<long>> reconstruct_tensors(std::vector<int> &flatten_snac) {
    auto size_in_two_hash = countElementsBetweenHashes(flatten_snac);
    if(size_in_two_hash != 7)
        throw std::runtime_error("elem between hash not 7");
    std::vector<std::vector<long>> snac_output(3);
    for (int i = 0; i < flatten_snac.size();) {
        snac_output[0].emplace_back((long)flatten_snac[i + 1]);

        snac_output[1].emplace_back((long)flatten_snac[i + 2]);
        snac_output[1].emplace_back((long)flatten_snac[i + 5]);

        snac_output[2].emplace_back((long)flatten_snac[i + 3]);
        snac_output[2].emplace_back((long)flatten_snac[i + 4]);
        snac_output[2].emplace_back((long)flatten_snac[i + 6]);
        snac_output[2].emplace_back((long)flatten_snac[i + 7]);

        i += 8;
    }
    return snac_output;
}

std::vector<int> reconscruct_snac(std::vector<std::vector<int>> &src_snac) {
    std::vector<std::vector<int>> src_snac_(7);
    for (int i = 0; i < 7; i++) {
        src_snac_[i] = std::vector<int>(src_snac[i].begin() + i+1, src_snac[i].end());
    }
    std::vector<int> snac_output;
    size_t last_size = src_snac_[src_snac_.size() - 1].size();


    for (int i = 0; i < last_size; i++) {
        snac_output.emplace_back(hash_flag);
        for (int j = 0; j < 7; j++) {
            snac_output.emplace_back(src_snac_[j][i]);
        }
    }
    return snac_output;
}

std::string A1_A2(tensor_info<float> &audio_feature,
                  tensor_info<long> &input_ids,
                  int length,
                  ONNXModel &adapter,
                  ONNXModel &gpt,
                  std::unique_ptr<tokenizers::Tokenizer> &tokenizer) {
#if 1
    auto tokenizer_list = generate_AA(audio_feature, input_ids, adapter, gpt,
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

#else
    std::vector<std::vector<int>> tokenizer_list = {
        {4097, 1425, 3450, 1953, 580, 3950, 1739, 1091, 1387, 1387, 3121, 1537, 1769, 1722, 3331, 499, 2483, 2382, 581, 3395, 705, 3403, 1109, 3325, 3230, 1210, 3989, 3069, 2816, 3507, 2170, 1349, 3340, 153, 3634, 1785, 2833, 3794, 2170, 1349, 3340, 153, 3598, 201, 1415, 957, 2049, 2998, 1841, 899, 1057, 1425, 3981, 4096, 4097, 4097, 4097, 4097, 4097},
        {4097, 4097, 2031, 3919, 1534, 1678, 2323, 2149, 1011, 2000, 3130, 3384, 3352, 3363, 3256, 665, 766, 2128, 3130, 86, 1994, 3044, 766, 2730, 1321, 3026, 2833, 4017, 1873, 1558, 1983, 2576, 2543, 468, 1742, 368, 1469, 1465, 2168, 1823, 2543, 3611, 1742, 2632, 3214, 3553, 2911, 178, 3457, 1290, 2095, 31, 1190, 2426, 4096, 4097, 4097, 4097, 4097},
        {4097, 4097, 4097, 2825, 436, 2699, 4000, 848, 2755, 560, 3622, 3010, 2763, 559, 3140, 2499, 3431, 212, 1895, 866, 1308, 2488, 1768, 1805, 405, 1337, 115, 2114, 3985, 351, 486, 1561, 3523, 668, 2244, 2222, 3144, 1516, 2806, 1436, 443, 2699, 2624, 3783, 1091, 2822, 986, 2466, 3262, 3738, 3421, 1459, 2570, 1479, 3909, 4096, 4097, 4097, 4097},
        {4097, 4097, 4097, 4097, 2111, 960, 3847, 556, 2189, 1819, 147, 528, 351, 3694, 94, 1628, 2701, 1378, 678, 734, 1332, 3823, 1978, 1061, 3846, 98, 1121, 2919, 506, 1271, 3491, 149, 2617, 1763, 2414, 487, 3796, 2228, 885, 3237, 1557, 858, 1893, 3872, 3796, 1104, 2093, 3088, 3867, 2098, 3127, 3117, 3561, 4068, 2569, 1532, 4096, 4097, 4097},
        {4097, 4097, 4097, 4097, 4097, 1144, 3396, 774, 976, 468, 3552, 3130, 3861, 1154, 2481, 3849, 1843, 3673, 3741, 1504, 2983, 667, 765, 1583, 580, 1139, 899, 863, 1960, 169, 582, 774, 3130, 1766, 1294, 927, 2585, 1666, 3620, 3129, 3186, 1766, 3474, 927, 2585, 2543, 28, 1885, 2693, 3517, 1084, 840, 2602, 707, 3397, 2112, 2609, 4096, 4097},
        {4097, 4097, 4097, 4097, 4097, 4097, 4085, 3320, 902, 3268, 2271, 2706, 1546, 1114, 4056, 2683, 729, 3330, 770, 1451, 1367, 2422, 3714, 663, 802, 1602, 3306, 2690, 550, 1911, 3866, 2501, 1221, 1012, 1246, 2455, 809, 1076, 1655, 139, 182, 3417, 2182, 2378, 674, 1434, 2357, 1709, 3218, 2641, 3246, 4071, 2230, 2581, 2929, 114, 585, 3602, 4096},
        {4097, 4097, 4097, 4097, 4097, 4097, 4097, 3204, 4074, 3326, 2321, 2944, 1826, 1331, 3742, 844, 1878, 480, 2589, 3322, 2548, 2752, 1099, 2793, 1218, 317, 1153, 1200, 3539, 2933, 2537, 579, 2338, 3941, 1221, 759, 3000, 3321, 747, 307, 1142, 2415, 3056, 2590, 93, 527, 3496, 1471, 2446, 601, 408, 2707, 1790, 224, 3875, 229, 1291, 3945, 1855},
        {40, 1513, 944, 614, 264, 829, 11, 714, 358, 2776, 1588, 311, 1492, 498, 448, 894, 4755, 476, 4755, 498, 614, 0, 151936, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937}
    };
#endif

#if DUMP_WAV
    auto audio_list = reconscruct_snac(tokenizer_list);
    auto audio = reconstruct_tensors(audio_list);

    // process 3 audio into 1 for single dynamic axis.
    std::vector<long> audio_(audio[0].begin(), audio[0].end());
    audio_.insert(audio_.end(), audio[1].begin(), audio[1].end());
    audio_.insert(audio_.end(), audio[2].begin(), audio[2].end());
    tensor_info<long> snac_input_tensor{.data = audio_, .shape = {1, (int)audio_.size()}};

    std::vector<Value> inputs;
    auto snac_input = Input<long>(snac_input_tensor.shape, snac.runtime_manager_);
    auto snac_input_ptr = snac_input.GetTensorMutableData<long>();
    std::memcpy(snac_input_ptr, snac_input_tensor.data.data(), snac_input_tensor.data.size() * sizeof(long));
    inputs.emplace_back(std::move(snac_input));

    auto snac_output = snac.onForward(inputs);

    auto audio_hat = snac.get_result_vector<float>(snac_output, 0);

    std::string save_path = "../data/output.wav";
    save_audio(save_path, audio_hat.data, 24000);
#endif

    auto vec = tokenizer_list.back();
    size_t size = vec.size();
    auto it = std::find(vec.begin(), vec.end(), text_vocabsize);
    if (it != vec.end()) {
        size = std::distance(vec.begin(), it) + 1;
    }
    vec.resize(size);
    auto text = tokenizer->Decode(vec);
    return strip(text);
}

std::pair<tensor_info<float>, tensor_info<long>>
generate_input_ids(ONNXModel &model, tensor_info<float> &mel, int length,
                   int step,
                   int special_token_a, int special_token_t) {

    mel.shape = {1, mel.shape[0], mel.shape[1]};
    auto audio_feature = model_run<float, float>(model, mel);
    std::vector<float> part_audio_of_length(audio_feature.data.begin(),
                                            audio_feature.data.begin() + length * audio_feature.shape[2]);
    audio_feature.data = part_audio_of_length;
    audio_feature.shape = {length, audio_feature.shape[2]};

    std::vector<std::vector<long>> input_ids(8);
    for (int i = 0; i < 7; ++i) {
        input_ids[i].push_back(_input_a + 152000 + i * 4160);
        input_ids[i].insert(input_ids[i].end(), length, _pad_a + 152000 + i * 4160);
        input_ids[i].push_back(_eoa + +152000 + i * 4160);
        input_ids[i].push_back(special_token_a + +152000 + i * 4160);

    }
    input_ids[7].push_back(_input_t);
    input_ids[7].insert(input_ids[7].end(), length, _pad_t);
    input_ids[7].push_back(_eot);
    input_ids[7].push_back(special_token_t);

    std::vector<long> input_ids_;
    for (int i = 0; i < 8; i++) {
        input_ids_.insert(input_ids_.end(), input_ids[i].begin(), input_ids[i].end());
    }

    return {audio_feature, {.data=input_ids_, .shape={(long) input_ids.size(), (long) input_ids[0].size()}}};
}

#else
void dump_shape(const std::string &info, nncase::dims_t &shape)
{
    std::cout << info << ": [";
    for (auto s : shape)
        std::cout << s << " ";
    std::cout << "]" << std::endl;
}

std::tuple<std::vector<int>, int, tensor_info<float>, tensor_info<float>>
next_token_A1T2(NncaseModel &gpt, tensor_info<float> &input_embs_concat, tensor_info<long> &input_pos_t,
                tensor_info<float> &past_ks_tensor, tensor_info<float> &past_vs_tensor, int sub_step,
                float temperature, int top_k, float top_p) {
    std::vector<nncase::value_t> inputs;
    auto entry = gpt.entry();

    // input 0: input_embs
    auto type = entry->parameter_type(0).expect("parameter type out of index");
    auto ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    auto data_type = ts_type->dtype()->typecode();
    nncase::dims_t input_embs_shape(input_embs_concat.shape.begin(), input_embs_concat.shape.end());
    auto input_embs_tensor = nncase::runtime::host_runtime_tensor::create(data_type, input_embs_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto input_embs_buffer = input_embs_tensor->buffer().as_host().unwrap_or_throw();
    auto input_embs_mapped = input_embs_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto input_embs_ptr = input_embs_mapped.buffer().as_span<float>().data();
    std::memcpy(input_embs_ptr, input_embs_concat.data.data(), input_embs_concat.data.size() * sizeof(float));
    input_embs_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(input_embs_tensor);

    // input 1: past_keys
    type = entry->parameter_type(1).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t past_keys_shape(past_ks_tensor.shape.begin(), past_ks_tensor.shape.end());
    auto past_keys_tensor = nncase::runtime::host_runtime_tensor::create(data_type, past_keys_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto past_keys_buffer = past_keys_tensor->buffer().as_host().unwrap_or_throw();
    auto past_keys_mapped = past_keys_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto past_keys_ptr = past_keys_mapped.buffer().as_span<float>().data();
    std::memcpy(past_keys_ptr, past_ks_tensor.data.data(), past_ks_tensor.data.size() * sizeof(float));
    past_keys_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(past_keys_tensor);

    // input 2: past_values
    type = entry->parameter_type(2).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t past_values_shape(past_vs_tensor.shape.begin(), past_vs_tensor.shape.end());
    auto past_values_tensor = nncase::runtime::host_runtime_tensor::create(data_type, past_values_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto past_values_buffer = past_values_tensor->buffer().as_host().unwrap_or_throw();
    auto past_values_mapped = past_values_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto past_values_ptr = past_values_mapped.buffer().as_span<float>().data();
    std::memcpy(past_values_ptr, past_vs_tensor.data.data(), past_vs_tensor.data.size() * sizeof(float));
    past_values_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(past_values_tensor);

    // input 3: input_pos
    type = entry->parameter_type(3).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t input_pos_shape(input_pos_t.shape.begin(), input_pos_t.shape.end());
    auto input_pos_tensor = nncase::runtime::host_runtime_tensor::create(data_type, input_pos_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto input_pos_buffer = input_pos_tensor->buffer().as_host().unwrap_or_throw();
    auto input_pos_mapped = input_pos_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto input_pos_ptr = input_pos_mapped.buffer().as_span<long>().data();
    std::memcpy(input_pos_ptr, input_pos_t.data.data(), input_pos_t.data.size() * sizeof(long));
    input_pos_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(input_pos_tensor);

    // run
    nncase::value_t outs;
    {
        ScopedTiming st("lit_gpt invoke");
        outs = gpt.run(inputs);
    }
    auto outputs = outs.as<nncase::tuple>().unwrap();

    // get output 0
    auto logits_a_tensor = outputs->fields()[0].as<nncase::tensor>().unwrap_or_throw();
    auto logits_a_buffer = logits_a_tensor->buffer().as_host().unwrap_or_throw();
    auto logits_a_mapped = logits_a_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto logits_a_data = logits_a_mapped.buffer().as_span<float>();
    std::vector<float> logits_a_v(logits_a_data.begin(), logits_a_data.end());
    std::vector<long> logits_a_shape(logits_a_tensor->shape().begin(), logits_a_tensor->shape().end());
    tensor_info<float> logits_a = {logits_a_v, logits_a_shape };

    // get output 1
    auto logits_t_tensor = outputs->fields()[1].as<nncase::tensor>().unwrap_or_throw();
    auto logits_t_buffer = logits_t_tensor->buffer().as_host().unwrap_or_throw();
    auto logits_t_mapped = logits_t_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto logits_t_data = logits_t_mapped.buffer().as_span<float>();
    std::vector<float> logits_t_v(logits_t_data.begin(), logits_t_data.end());
    std::vector<long> logits_t_shape(logits_t_tensor->shape().begin(), logits_t_tensor->shape().end());
    tensor_info<float> logit_t = {logits_t_v, logits_t_shape };

    // get output 2
    auto next_ks_tensor = outputs->fields()[2].as<nncase::tensor>().unwrap_or_throw();
    auto next_ks_buffer = next_ks_tensor->buffer().as_host().unwrap_or_throw();
    auto next_ks_mapped = next_ks_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto next_ks_data = next_ks_mapped.buffer().as_span<float>();
    std::vector<float> next_ks_v(next_ks_data.begin(), next_ks_data.end());
    std::vector<long> next_ks_shape(next_ks_tensor->shape().begin(), next_ks_tensor->shape().end());
    tensor_info<float> next_ks = {next_ks_v, next_ks_shape};

    // get output 3
    auto next_vs_tensor = outputs->fields()[3].as<nncase::tensor>().unwrap_or_throw();
    auto next_vs_buffer = next_vs_tensor->buffer().as_host().unwrap_or_throw();
    auto next_vs_mapped = next_vs_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto next_vs_data = next_vs_mapped.buffer().as_span<float>();
    std::vector<float> next_vs_v(next_vs_data.begin(), next_vs_data.end());
    std::vector<long> next_vs_shape(next_vs_tensor->shape().begin(), next_vs_tensor->shape().end());
    tensor_info<float> next_vs = {next_vs_v, next_vs_shape};

    // postprocess
    std::vector<int> next_audio_tokens;
    for (int i = 0; i < logits_a.shape[0]; i++) {
        std::vector<float> logits_a_i_data(
                std::vector<float>(logits_a.data.data() + i * logits_a.shape[2] * logits_a.shape[3],
                                   logits_a.data.data() + (i + 1) * logits_a.shape[2] * logits_a.shape[3]));
        tensor_info<float> logits_a_i{.data = logits_a_i_data, .shape = {1, logits_a.shape[2], logits_a.shape[3]}};
        auto next_a = sample(logits_a_i, temperature, top_k, top_p);

        next_audio_tokens.emplace_back(next_a);
    }
    auto next_t = sample(logit_t, temperature, top_k, top_p);
    return {next_audio_tokens, next_t, next_ks, next_vs};
}


std::vector<std::vector<int>>
generate_AA(tensor_info<float> &audio_feature, tensor_info<long> &input_ids,
            NncaseModel &adapter, NncaseModel &gpt,
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
    // std::cout << "nncase generate_AA is entering: audio shape = " << audio_feature.shape[0] << " " << audio_feature.shape[1] << std::endl;
    auto T = input_ids.shape[1];
    std::vector<std::vector<int>> outputs(8);
    // write_binary_file("adapter_input_audio_feature.bin", reinterpret_cast<char *>(audio_feature.data.data()), audio_feature.data.size() * sizeof(float));

    // 1. adapter
    std::vector<long> audio_shape = {1, audio_feature.shape[0], audio_feature.shape[1]};
    std::vector<nncase::value_t> inputs;
    auto entry = adapter.entry();
    auto type = entry->parameter_type(0).expect("parameter type out of index");
    auto ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    auto data_type = ts_type->dtype()->typecode();

    nncase::dims_t shape(audio_shape.begin(), audio_shape.end());
    auto audio_features_tensor = nncase::runtime::host_runtime_tensor::create(data_type, shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto audio_features_buffer = audio_features_tensor->buffer().as_host().unwrap_or_throw();
    auto audio_features_mapped = audio_features_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto adapter_ptr = audio_features_mapped.buffer().as_span<float>().data();
    // std::cout << "audio_feature.data.size() " << audio_feature.data.size() << std::endl;
    memcpy(reinterpret_cast<void *>(adapter_ptr), reinterpret_cast<void *>(audio_feature.data.data()), audio_feature.data.size() * sizeof(float));

    audio_features_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(audio_features_tensor);
    // std::cout << "adapter inputs done." << std::endl;
    nncase::value_t output;
    {
        ScopedTiming st("adapter invoke");
        output = adapter.run(inputs);
    }

    // std::cout << "adapter run done." << std::endl;
    auto audio_embs_tensor = output.as<nncase::tensor>().unwrap_or_throw();
    auto audio_embs_buffer = audio_embs_tensor->buffer().as_host().unwrap_or_throw();
    auto audio_embs_mapped = audio_embs_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto audio_embs_data = audio_embs_mapped.buffer().as_span<float>();
    std::vector<float> audio_embs_v(audio_embs_data.begin(), audio_embs_data.end());
    // write_binary_file("adapter_nncase_audio_embs.bin", reinterpret_cast<char *>(audio_embs_v.data()), audio_embs_v.size() * sizeof(float));
    std::vector<long> audio_embs_shape(audio_embs_tensor->shape().begin(), audio_embs_tensor->shape().end());
    tensor_info<float> audio_embs = {audio_embs_v, audio_embs_shape};

    // 2. WTE
    input_ids.shape = {input_ids.shape[0], 1, input_ids.shape[1]};
    auto input_embs = wte_get_data(input_ids);

    // 3. lit_gpt
    auto input_embs_concat = concat_feat(audio_embs, input_embs);

    std::vector<float> past_ks(0, 0);
    std::vector<float> past_vs(0, 0);
    std::vector<long> input_pos;
    for (int i = 0; i < T; i++)
        input_pos.emplace_back(i);

    tensor_info<float> past_ks_tensor{.data=past_ks, .shape={24, 1, 14, 0, 64}};
    tensor_info<float> past_vs_tensor{.data=past_vs, .shape={24, 1, 14, 0, 64}};
    tensor_info<long> input_pos_tensor{.data=input_pos, .shape={(long) input_pos.size()}};


    auto [tokens_A, token_T, past_ks_, past_vs_] = next_token_A1T2(gpt, input_embs_concat, input_pos_tensor,
                                                                   past_ks_tensor, past_vs_tensor, 1, temperature,
                                                                   top_k,
                                                                   top_p);

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

        auto input_embs_loop_tensor = wte_get_data(input_ids_tensor);

        tensor_info<long> input_pos_loop_tensor{.data=input_pos, .shape={(long) input_pos.size()}};
//        std::tuple<std::vector<int>, int, tensor_info<float>, tensor_info<float>>
        auto [_tokens_A, _token_T, _past_ks_, _past_vs_] = next_token_A1T2(gpt,
                                                                           input_embs_loop_tensor,
                                                                           input_pos_loop_tensor,
                                                                           past_ks_,
                                                                           past_vs_,
                                                                           sub_step,
                                                                           temperature,
                                                                           top_k,
                                                                           top_p);
        tokens_A = _tokens_A;
        token_T = _token_T;
        past_ks_ = _past_ks_;
        past_vs_ = _past_vs_;

        if (text_end)
            token_T = pad_id_t;
        if ((int) tokens_A[tokens_A.size() - 1] == eos_id_a)
            break;
        if (token_T == eos_id_t)
            text_end = true;

        for (int i = 0; i < 7; i++) {
            outputs[i].emplace_back(tokens_A[i]);
        }
        outputs[7].emplace_back(token_T);
        input_pos[0] += 1;

        // for (int i = 0; i < 8; i++)
        // {
        //     std::cout << "output: [" << i << "] ";
        //     for (auto t : outputs[i])
        //         std::cout << t << " ";
        //     std::cout<<std::endl;
        // }
    }

    return outputs;
}

int countElementsBetweenHashes(const std::vector<int> &lst) {
    try {
        // Find the index of the first '#'
        auto first_index = std::find(lst.begin(), lst.end(), hash_flag);

        if (first_index == lst.end()) {
            throw std::invalid_argument("No '#' found in the list.");
        }

        // Find the index of the second '#' after the first
        auto second_index = std::find(first_index + 1, lst.end(), hash_flag);

        if (second_index == lst.end()) {
            throw std::invalid_argument("Only one '#' found in the list.");
        }

        // Calculate the number of elements between the two indices
        return std::distance(first_index, second_index) - 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1; // 返回-1表示发生错误
    }
}

std::vector<std::vector<long>> reconstruct_tensors(std::vector<int> &flatten_snac) {
    auto size_in_two_hash = countElementsBetweenHashes(flatten_snac);
    if(size_in_two_hash != 7)
        throw std::runtime_error("elem between hash not 7");
    std::vector<std::vector<long>> snac_output(3);
    for (int i = 0; i < flatten_snac.size();) {
        snac_output[0].emplace_back((long)flatten_snac[i + 1]);

        snac_output[1].emplace_back((long)flatten_snac[i + 2]);
        snac_output[1].emplace_back((long)flatten_snac[i + 5]);

        snac_output[2].emplace_back((long)flatten_snac[i + 3]);
        snac_output[2].emplace_back((long)flatten_snac[i + 4]);
        snac_output[2].emplace_back((long)flatten_snac[i + 6]);
        snac_output[2].emplace_back((long)flatten_snac[i + 7]);

        i += 8;
    }
    return snac_output;
}

std::vector<int> reconscruct_snac(std::vector<std::vector<int>> &src_snac) {
    std::vector<std::vector<int>> src_snac_(7);
    for (int i = 0; i < 7; i++) {
        src_snac_[i] = std::vector<int>(src_snac[i].begin() + i+1, src_snac[i].end());
    }
    std::vector<int> snac_output;
    size_t last_size = src_snac_[src_snac_.size() - 1].size();


    for (int i = 0; i < last_size; i++) {
        snac_output.emplace_back(hash_flag);
        for (int j = 0; j < 7; j++) {
            snac_output.emplace_back(src_snac_[j][i]);
        }
    }
    return snac_output;
}

std::string A1_A2(tensor_info<float> &audio_feature,
                  tensor_info<int64_t> &input_ids,
                  int length,
                  NncaseModel &adapter,
                  NncaseModel &gpt,
                  std::unique_ptr<tokenizers::Tokenizer> &tokenizer) {
#if 1
    auto tokenizer_list = generate_AA(audio_feature, input_ids, adapter, gpt,
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

#else
    std::vector<std::vector<int>> tokenizer_list = {
        {4097, 1425, 3450, 1953, 580, 3950, 1739, 1091, 1387, 1387, 3121, 1537, 1769, 1722, 3331, 499, 2483, 2382, 581, 3395, 705, 3403, 1109, 3325, 3230, 1210, 3989, 3069, 2816, 3507, 2170, 1349, 3340, 153, 3634, 1785, 2833, 3794, 2170, 1349, 3340, 153, 3598, 201, 1415, 957, 2049, 2998, 1841, 899, 1057, 1425, 3981, 4096, 4097, 4097, 4097, 4097, 4097},
        {4097, 4097, 2031, 3919, 1534, 1678, 2323, 2149, 1011, 2000, 3130, 3384, 3352, 3363, 3256, 665, 766, 2128, 3130, 86, 1994, 3044, 766, 2730, 1321, 3026, 2833, 4017, 1873, 1558, 1983, 2576, 2543, 468, 1742, 368, 1469, 1465, 2168, 1823, 2543, 3611, 1742, 2632, 3214, 3553, 2911, 178, 3457, 1290, 2095, 31, 1190, 2426, 4096, 4097, 4097, 4097, 4097},
        {4097, 4097, 4097, 2825, 436, 2699, 4000, 848, 2755, 560, 3622, 3010, 2763, 559, 3140, 2499, 3431, 212, 1895, 866, 1308, 2488, 1768, 1805, 405, 1337, 115, 2114, 3985, 351, 486, 1561, 3523, 668, 2244, 2222, 3144, 1516, 2806, 1436, 443, 2699, 2624, 3783, 1091, 2822, 986, 2466, 3262, 3738, 3421, 1459, 2570, 1479, 3909, 4096, 4097, 4097, 4097},
        {4097, 4097, 4097, 4097, 2111, 960, 3847, 556, 2189, 1819, 147, 528, 351, 3694, 94, 1628, 2701, 1378, 678, 734, 1332, 3823, 1978, 1061, 3846, 98, 1121, 2919, 506, 1271, 3491, 149, 2617, 1763, 2414, 487, 3796, 2228, 885, 3237, 1557, 858, 1893, 3872, 3796, 1104, 2093, 3088, 3867, 2098, 3127, 3117, 3561, 4068, 2569, 1532, 4096, 4097, 4097},
        {4097, 4097, 4097, 4097, 4097, 1144, 3396, 774, 976, 468, 3552, 3130, 3861, 1154, 2481, 3849, 1843, 3673, 3741, 1504, 2983, 667, 765, 1583, 580, 1139, 899, 863, 1960, 169, 582, 774, 3130, 1766, 1294, 927, 2585, 1666, 3620, 3129, 3186, 1766, 3474, 927, 2585, 2543, 28, 1885, 2693, 3517, 1084, 840, 2602, 707, 3397, 2112, 2609, 4096, 4097},
        {4097, 4097, 4097, 4097, 4097, 4097, 4085, 3320, 902, 3268, 2271, 2706, 1546, 1114, 4056, 2683, 729, 3330, 770, 1451, 1367, 2422, 3714, 663, 802, 1602, 3306, 2690, 550, 1911, 3866, 2501, 1221, 1012, 1246, 2455, 809, 1076, 1655, 139, 182, 3417, 2182, 2378, 674, 1434, 2357, 1709, 3218, 2641, 3246, 4071, 2230, 2581, 2929, 114, 585, 3602, 4096},
        {4097, 4097, 4097, 4097, 4097, 4097, 4097, 3204, 4074, 3326, 2321, 2944, 1826, 1331, 3742, 844, 1878, 480, 2589, 3322, 2548, 2752, 1099, 2793, 1218, 317, 1153, 1200, 3539, 2933, 2537, 579, 2338, 3941, 1221, 759, 3000, 3321, 747, 307, 1142, 2415, 3056, 2590, 93, 527, 3496, 1471, 2446, 601, 408, 2707, 1790, 224, 3875, 229, 1291, 3945, 1855},
        {40, 1513, 944, 614, 264, 829, 11, 714, 358, 2776, 1588, 311, 1492, 498, 448, 894, 4755, 476, 4755, 498, 614, 0, 151936, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937, 151937}
    };
#endif

#if DUMP_WAV
    auto audio_list = reconscruct_snac(tokenizer_list);
    auto audio = reconstruct_tensors(audio_list);

    // process 3 audio into 1 for single dynamic axis.
    std::vector<long> audio_(audio[0].begin(), audio[0].end());
    audio_.insert(audio_.end(), audio[1].begin(), audio[1].end());
    audio_.insert(audio_.end(), audio[2].begin(), audio[2].end());
    tensor_info<long> snac_input_tensor{.data = audio_, .shape = {1, (int)audio_.size()}};

    std::vector<Value> inputs;
    auto snac_input = Input<long>(snac_input_tensor.shape, snac.runtime_manager_);
    auto snac_input_ptr = snac_input.GetTensorMutableData<long>();
    std::memcpy(snac_input_ptr, snac_input_tensor.data.data(), snac_input_tensor.data.size() * sizeof(long));
    inputs.emplace_back(std::move(snac_input));

    auto snac_output = snac.onForward(inputs);

    auto audio_hat = snac.get_result_vector<float>(snac_output, 0);

    std::string save_path = "../data/output.wav";
    save_audio(save_path, audio_hat.data, 24000);
#endif

    auto vec = tokenizer_list.back();
    size_t size = vec.size();
    auto it = std::find(vec.begin(), vec.end(), text_vocabsize);
    if (it != vec.end()) {
        size = std::distance(vec.begin(), it) + 1;
    }
    vec.resize(size);
    auto text = tokenizer->Decode(vec);
    return strip(text);
}

std::pair<tensor_info<float>, tensor_info<long>>
generate_input_ids(NncaseModel &model, tensor_info<float> &mel, int length,
                   int step,
                   int special_token_a, int special_token_t) {
    std::vector<nncase::value_t> inputs;
    auto entry = model.entry();
    auto type = entry->parameter_type(0).expect("parameter type out of index");
    auto ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    auto data_type = ts_type->dtype()->typecode();

    nncase::dims_t shape(mel.shape.begin(), mel.shape.end());
    auto mel_tensor = nncase::runtime::host_runtime_tensor::create(data_type, shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto mel_buffer = mel_tensor->buffer().as_host().unwrap_or_throw();
    auto mel_mapped = mel_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto ptr = mel_mapped.buffer().as_span<float>().data();
    memcpy(reinterpret_cast<void*>(ptr), reinterpret_cast<void*>(mel.data.data()), mel.data.size() * sizeof(float));

    mel_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(mel_tensor);
    nncase::value_t output;
    {
        ScopedTiming st("whisper invoke");
        output = model.run(inputs);
    }

    auto af_tensor = output.as<nncase::tensor>().unwrap_or_throw();
    auto af_buffer = af_tensor->buffer().as_host().unwrap_or_throw();
    auto af_mapped = af_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto output_data = af_mapped.buffer().as_span<float>();
    auto output_shape = af_tensor->shape();
    auto size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    write_binary_file("whisper_audio_feature_nncase.bin", reinterpret_cast<char *>(output_data.data()), size * sizeof(float));

    tensor_info<float> audio_feature;
    std::vector<float> part_audio_of_length(output_data.begin(),
                                            output_data.begin() + length * output_shape[2]);
    audio_feature.data = part_audio_of_length;
    audio_feature.shape = {length, output_shape[2]};
    std::cout << "audio_feature shape: " << length << " " << output_shape[2] << std::endl;
    write_binary_file("whisper_audio_feature_nncase_postprocess.bin", reinterpret_cast<char *>(audio_feature.data.data()), audio_feature.data.size() * sizeof(float));

    std::vector<std::vector<int64_t>> input_ids(8);
    for (int i = 0; i < 7; ++i) {
        input_ids[i].push_back(_input_a + 152000 + i * 4160);
        input_ids[i].insert(input_ids[i].end(), length, _pad_a + 152000 + i * 4160);
        input_ids[i].push_back(_eoa + +152000 + i * 4160);
        input_ids[i].push_back(special_token_a + +152000 + i * 4160);
    }
    input_ids[7].push_back(_input_t);
    input_ids[7].insert(input_ids[7].end(), length, _pad_t);
    input_ids[7].push_back(_eot);
    input_ids[7].push_back(special_token_t);

    std::vector<long> input_ids_;
    for (int i = 0; i < 8; i++) {
        input_ids_.insert(input_ids_.end(), input_ids[i].begin(), input_ids[i].end());
    }

    return {audio_feature, {.data=input_ids_, .shape={(long) input_ids.size(), (long) input_ids[0].size()}}};
}

#endif

std::string strip(const std::string &str, const std::string &chars) {
    // 查找第一个不在 chars 中的字符位置
    size_t start = str.find_first_not_of(chars);
    if (start == std::string::npos) return "";  // 全为需删除字符时返回空

    // 查找最后一个不在 chars 中的字符位置
    size_t end = str.find_last_not_of(chars);

    // 截取子字符串（含 start 和 end 的位置）
    return str.substr(start, end - start + 1);
}
