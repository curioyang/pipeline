//
// Created by curio on 2025/2/9.
//
#include "utils.h"
#include <complex>
#include <queue>
#include <utility> // for std::pair
#include <algorithm>


std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();

    // 检查是否可以进行矩阵乘法
    if (A_cols != B_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // 初始化结果矩阵
    std::vector<std::vector<float>> result(A_rows, std::vector<float>(B_cols, 0.0f));

    // 执行矩阵乘法
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


std::vector<float> reflectPad(const std::vector<float> &input, size_t padFront, size_t padBack) {
    // 计算填充后的新大小
    size_t newSize = input.size() + padFront + padBack;
    std::vector<float> padded(newSize);

    // 复制原始数据到中间位置
    std::copy(input.begin(), input.end(), padded.begin() + padFront);

    // 反射填充前端
    for (size_t i = 0; i < padFront; ++i) {
        padded[i] = input[padFront - i - 1];
    }

    // 反射填充后端
    for (size_t i = 0; i < padBack; ++i) {
        padded[padded.size() - 1 - i] = input[input.size() - 1 - i];
    }

    return padded;
}

template<class T>
std::vector<std::pair<T, int> > topK(const std::vector<T> &arr, size_t k) {
    std::vector<std::pair<T, int> > result;

    if (k <= 0 || arr.empty()) {
        throw std::invalid_argument("Invalid k value");
    }

    std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int> >, std::greater<std::pair<T, int> > > minHeap;

    for (int i = 0; i < arr.size(); ++i) {
        T value = arr[i];
        int index = i;

        // Push the current element into the heap
        if (minHeap.size() < k) {
            minHeap.push({value, index});
        } else {
            // Compare with the smallest element in the heap
            if (value > minHeap.top().first) {
                minHeap.pop();
                minHeap.push({value, index});
            }
        }
    }

    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
}

template std::vector<std::pair<float, int>> topK<float>(const std::vector<float> &, size_t);

template<class T>
std::vector<T> softmax(const std::vector<T> &x) {
    std::vector<T> result;
    double max_x = x[0];

    for (double val: x) {
        if (val > max_x) max_x = val;
    }

    double sum = 0.0;
    for (double val: x) {
        sum += std::exp(val - max_x);
    }

    for (double val: x) {
        result.push_back((float) (std::exp(val - max_x) / sum));
    }

    return result;
}