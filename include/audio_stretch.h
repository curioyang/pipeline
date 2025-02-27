#include <rubberband/RubberBandStretcher.h>
#include <vector>

std::vector<float> timeStretchPitchMaintain(const std::vector<float> &input,
                                            double time_ratio,
                                            int sample_rate = 24000,
                                            int channels = 1)
{
    // 参数校验
    if (input.empty() || time_ratio <= 0.01 || time_ratio > 10.0)
    {
        return {};
    }

    // 初始化拉伸器（优化配置）
    const RubberBand::RubberBandStretcher::Options options =
        RubberBand::RubberBandStretcher::OptionEngineFiner |
        RubberBand::RubberBandStretcher::OptionProcessOffline |
        RubberBand::RubberBandStretcher::OptionChannelsTogether;

    RubberBand::RubberBandStretcher stretcher(
        sample_rate,
        channels,
        options,
        time_ratio, // 时间伸缩比率
        1.0         // 音调比率（1.0表示不变）
    );

    // 预分配输出缓冲区（估算大小+10%余量）
    std::vector<float> output;
    output.reserve(static_cast<size_t>(input.size() / time_ratio * 1.1));

    // 处理参数
    const size_t block_size = 2048; // 经过测试的最佳块大小
    size_t input_pos = 0;

    // 处理循环
    while (input_pos < input.size())
    {
        // 计算当前块的实际大小
        const size_t remaining = input.size() - input_pos;
        const size_t this_block = std::min(block_size, remaining);

        // 构造输入块指针（兼容多声道接口）
        const float *input_block[1] = {input.data() + input_pos};

        // 提交处理
        stretcher.process(input_block, this_block, false);
        input_pos += this_block;

        // 获取可用输出
        while (stretcher.available() > 0)
        {
            const size_t avail = stretcher.available();

            // 准备输出缓冲区
            const size_t output_current = output.size();
            output.resize(output_current + avail);

            // 构造单声道指针数组
            float *output_ptr[1] = {output.data() + output_current};

            // 获取数据（关键调用）
            const size_t retrieved = stretcher.retrieve(output_ptr, avail);

            // 调整实际大小
            output.resize(output_current + retrieved);
        }
    }

    // 结束处理（冲刷内部缓存）
    stretcher.process(nullptr, 0, true);

    // 获取剩余数据
    while (stretcher.available() > 0)
    {
        const size_t avail = stretcher.available();
        const size_t output_current = output.size();
        output.resize(output_current + avail);

        float *output_ptr[1] = {output.data() + output_current};
        const size_t retrieved = stretcher.retrieve(output_ptr, avail);
        output.resize(output_current + retrieved);
    }

    // 优化内存占用
    output.shrink_to_fit();
    return output;
}