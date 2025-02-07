//
// Created by curio on 2025/2/7.
//

#ifndef WAV2WAV_UTILS_H
#define WAV2WAV_UTILS_H
#include <iostream>
#include <vector>
#include <cmath>

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
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


std::vector<float> reflectPad(const std::vector<float>& input, size_t padFront, size_t padBack) {
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
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix) {
    // 获取原矩阵的行数和列数
    size_t rows = matrix.size();
    if (rows == 0) return {}; // 如果原矩阵为空，直接返回空矩阵
    size_t cols = matrix[0].size();

    // 创建一个新的矩阵，其行数为原矩阵的列数，列数为原矩阵的行数
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    // 对于每个元素进行转置操作
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

#endif //WAV2WAV_UTILS_H
