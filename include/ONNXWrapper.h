//
// Created by Curio on 2/17/25.
//
#pragma once

// #ifdef BUILD_ONNX
#include "common.h"
#include "timer.h"
#include <numeric>
#include <onnxruntime_cxx_api.h>

using namespace Ort;

namespace omni_onnx
{
    class RuntimeManager
    {
    public:
        explicit RuntimeManager(const char *name)
        {
            name_ = name;
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, name);
            options_ = std::make_unique<Ort::SessionOptions>();
            options_->SetIntraOpNumThreads(6);
            options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        }

        ~RuntimeManager() = default;

        [[nodiscard]] const Ort::Env &env() const
        {
            return *env_;
        }

        [[nodiscard]] const Ort::SessionOptions &options() const
        {
            return *options_;
        }

        [[nodiscard]] const Ort::AllocatorWithDefaultOptions &allocator() const
        {
            return *allocator_;
        }

        [[nodiscard]] std::string name() const
        {
            return name_;
        }

    private:
        std::string name_;
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::SessionOptions> options_;
        std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    };

    class ONNXModel
    {
    public:
        ONNXModel(const std::shared_ptr<RuntimeManager> &runtime, const std::string &path)
            : runtime_manager_(runtime)
        {
            session_ = std::make_unique<Ort::Session>(runtime->env(), path.c_str(), runtime->options());
            input_count_ = session_->GetInputCount();
            output_count_ = session_->GetOutputCount();
            for (int i = 0; i < input_count_; i++)
            {
                input_strs_.push_back(session_->GetInputNameAllocated(i, runtime->allocator()));
                input_names_.push_back(input_strs_[i].get());
            }
            for (int i = 0; i < output_count_; i++)
            {
                output_strs_.push_back(session_->GetOutputNameAllocated(i, runtime->allocator()));
                output_names_.push_back(output_strs_[i].get());
            }
        }

        void onForward()
        {
            Timer timer(runtime_manager_->name());
            this->results_ = session_->Run(Ort::RunOptions{nullptr},
                                           input_names_.data(), inputs_.data(), inputs_.size(),
                                           output_names_.data(), output_names_.size());
            inputs_.clear();
        }

        template <class T>
        tensor_info<T> get_result_vector(int idx)
        {
            auto &it = this->results_[idx];
            auto data_tmp = it.GetTensorMutableData<T>();
            auto shape = it.GetTensorTypeAndShapeInfo().GetShape();
            auto all_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
            std::vector<T> result(data_tmp, data_tmp + all_size);
            return {.data = result, .shape = shape};
        }

        Value get_result_tensor(int idx)
        {
            auto &it = this->results_[idx];
            return std::move(it);
        }

        template <class T>
        void set_input_tensor(tensor_info<T> &tensor, size_t idx)
        {
            inputs_.emplace_back(Value::CreateTensor<T>(runtime_manager_.get()->allocator().GetInfo(), tensor.data.data(), tensor.data.size(), tensor.shape.data(), tensor.shape.size()));
        }

        void set_input_tensor(Value &tensor, size_t idx)
        {
            inputs_.emplace_back(std::move(tensor));
        }

        std::pair<std::vector<long>, size_t> get_output_shape(int idx)
        {
            auto shape = session_.get()->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
            auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
            return {shape, size};
        }

        std::pair<std::vector<long>, size_t> get_input_shape(int idx)
        {
            auto shape = session_.get()->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
            auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
            return {shape, size};
        }

    private:
        std::shared_ptr<RuntimeManager> runtime_manager_;
        std::unique_ptr<Ort::Session> session_;
        size_t input_count_, output_count_;
        std::vector<AllocatedStringPtr> input_strs_, output_strs_;
        std::vector<const char *> input_names_, output_names_;
        std::vector<Value> results_;
        std::vector<Value> inputs_;
    };

}

// #endif