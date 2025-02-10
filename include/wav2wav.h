#include <onnxruntime_cxx_api.h>
#include <numeric>
#include <vector>
#include <cmath>

using namespace Ort;

constexpr int padded_audio_vocab = 4096 + 64;

constexpr int text_vocabsize = 151936;
constexpr int text_specialtokens = 64;
constexpr int audio_vocabsize = 4096;
constexpr int audio_specialtokens = 64;

constexpr int padded_text_vocabsize = text_vocabsize + text_specialtokens;
constexpr int padded_audio_vocabsize = audio_vocabsize + audio_specialtokens;

constexpr int _eot = text_vocabsize;
constexpr int _pad_t = text_vocabsize + 1;
constexpr int _input_t = text_vocabsize + 2;
constexpr int _answer_t = text_vocabsize + 3;
constexpr int _asr = text_vocabsize + 4;

constexpr int _eoa = audio_vocabsize;
constexpr int _pad_a = audio_vocabsize + 1;
constexpr int _input_a = audio_vocabsize + 2;
constexpr int _answer_a = audio_vocabsize + 3;
constexpr int _split = audio_vocabsize + 4;
constexpr int _image = audio_vocabsize + 5;
constexpr int _eoimage = audio_vocabsize + 6;


class RuntimeManager {
public:
    explicit RuntimeManager(const char *name) {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, name);
        options_ = std::make_unique<Ort::SessionOptions>();
        options_->SetIntraOpNumThreads(1);
        options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    }

    ~RuntimeManager() = default;

    [[nodiscard]] const Ort::Env &env() const {
        return *env_;
    }

    [[nodiscard]] const Ort::SessionOptions &options() const {
        return *options_;
    }

    [[nodiscard]] const Ort::AllocatorWithDefaultOptions &allocator() const {
        return *allocator_;
    }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> options_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
};

class ONNXModel {
public:
    ONNXModel(const std::shared_ptr<RuntimeManager> &runtime, const std::string &path)
            : runtime_manager_(runtime) {
        session_ = std::make_unique<Ort::Session>(runtime->env(), path.c_str(), runtime->options());
        input_count_ = session_->GetInputCount();
        output_count_ = session_->GetOutputCount();
        for (int i = 0; i < input_count_; i++) {
            input_strs_.push_back(session_->GetInputNameAllocated(i, runtime->allocator()));
            input_names_.push_back(input_strs_[i].get());
        }
        for (int i = 0; i < output_count_; i++) {
            output_strs_.push_back(session_->GetOutputNameAllocated(i, runtime->allocator()));
            output_names_.push_back(output_strs_[i].get());
        }
    }

    std::vector<Value> onForward(const std::vector<Value> &inputs) {
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names_.data(), inputs.data(), inputs.size(),
                                     output_names_.data(), output_names_.size());
        return outputs;
    }

    template<class T>
    auto get_result_vector(std::vector<Value> &data, int idx) {
        std::vector<long> result_shape;
        auto &it = data[idx];
        auto t = it.GetTensorTypeAndShapeInfo().GetElementType();

        auto data_tmp = it.GetTensorMutableData<T>();
        auto s = it.GetTensorTypeAndShapeInfo().GetShape();
        auto k = std::accumulate(s.begin(), s.end(), 1, std::multiplies<>());
        std::vector<T> result(data_tmp, data_tmp + k);
        return make_tuple(result, s);
    }

    std::shared_ptr<RuntimeManager> runtime_manager_;
private:
    std::unique_ptr<Ort::Session> session_;
    size_t input_count_, output_count_;
    std::vector<AllocatedStringPtr> input_strs_, output_strs_;
    std::vector<const char *> input_names_, output_names_;
};

struct GenerationResult {
    std::vector<int> audio_tokens;
    std::vector<int> text_tokens;
};

template<typename T>
static Value Input(const std::vector<int> &shape, const std::shared_ptr<RuntimeManager> &rtmgr) {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    return Value::CreateTensor<T>(rtmgr->allocator(), shape_int64.data(), shape_int64.size());
}

/*GenerationResult*/void A1_A2(std::vector<std::vector<float>> &audio_feature,
                       std::vector<std::vector<int64_t>> &input_ids,
                       int length,
                       ONNXModel &adapter,
                       ONNXModel &wte,
                       ONNXModel &gpt/*, Tokenizer& tokenizer*/);


//std::pair<std::vector<float>, int> load_audio(const std::string& path);


std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>>
generate_input_ids(ONNXModel &model, std::vector<std::vector<float>> &mel, int length,
                   int step = 0,
                   int special_token_a = _answer_a, int special_token_t = _answer_t);

#include <algorithm>
#include <random>

//int sample(const float* logits, int size, float temp, int top_k, float top_p);
