//
// Created by Curio on 2/17/25.
//
#ifndef ONNX

#pragma once

#include "common.h"
#include "timer.h"
#include <fstream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

class NNCASEModel
{
public:
    NNCASEModel(std::string &path, const char *name)
        : name_(name)
    {
        std::ifstream ifs(path, std::ios::binary);
        model_.load_model(ifs).expect("Invalid kmodel");
        entry_function_ = model_.entry_function().unwrap_or_throw();
        // inputs_.resize(entry_function_->parameters_size());
    }

    void onForward()
    {
        Timer timer(name_);
        results_ = entry_function_->invoke(inputs_)
                       .unwrap_or_throw()
                       .as<nncase::tuple>()
                       .unwrap_or_throw();
        inputs_.clear();
    }

    template <class T>
    tensor_info<T> get_result_vector(int idx)
    {
        auto tensor_ = this->results_->fields()[idx].as<nncase::tensor>().unwrap_or_throw();
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape_ = tensor_->shape();
        std::vector<T> result((T *)data, (T *)data + compute_size(tensor_));
        std::vector<long> shape(shape_.begin(), shape_.end());
        return {.data = result, .shape = shape};
    }

    nncase::value_t get_result_tensor(int idx)
    {
        auto tensor = this->results_->fields()[idx];
        return std::move(tensor);
    }

    template <class T>
    void set_input_tensor(tensor_info<T> &tensor, size_t idx)
    {
        auto type = entry_function_->parameter_type(idx).expect("parameter type out of index");
        auto ts_type = type.as<tensor_type>().expect("input is not a tensor type");
        dims_t shape = ts_type->shape().as_fixed().unwrap();
        auto data_type = ts_type->dtype()->typecode();

        auto input = host_runtime_tensor::create(data_type, shape, {(gsl::byte *)tensor.data.data(), (size_t)tensor.data.size() * sizeof(T)}, true, hrt::pool_shared).expect("cannot create input tensor");
        hrt::sync(input, sync_op_t::sync_write_back, true).unwrap();
        inputs_.emplace_back(input.impl());
    }

    void set_input_tensor(nncase::value_t &tensor, size_t idx)
    {
        inputs_.emplace_back(tensor);
    }

private:
    interpreter model_;
    std::vector<value_t> inputs_;
    nncase::tuple results_;
    runtime_function *entry_function_;
    std::string name_;
};

template <typename T>
static nncase::tensor _Input(const std::vector<int> &shape)
{
    nncase::dims_t shape_int64(shape.begin(), shape.end());
    return nncase::runtime::hrt::create(
               std::is_same_v<T, float> ? nncase::dt_float32 : nncase::dt_int32,
               shape_int64, nncase::runtime::host_runtime_tensor::pool_shared)
        .unwrap_or_throw()
        .impl();
}
#endif