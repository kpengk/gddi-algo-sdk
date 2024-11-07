/**
 * @file safety_belt_algo.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 安全带佩戴以及挂钩检测
 * @version 1.0.0
 * @date 2024-10-17
 * 
 * @copyright Copyright (c) 2024 by GDDI
 * 
 */

#pragma once

#include "struct_def.h"
#include <api/infer_api.h>
#include <core/result_def.h>
#include <opencv2/core/mat.hpp>
#include <set>
#include <vector>

namespace gddi {

struct SafetyBeltAlgoConfig {
    uint32_t delay_time{3};    // 延迟时间
    float light_threshold{0.3};// 灯光统计阈值

    uint32_t statistics_time{5};     // 统计时间
    float safety_belt_threshold{0.5};// 安全带统计阈值
};

class SafetyBeltAlgo {
public:
    SafetyBeltAlgo(const SafetyBeltAlgoConfig &config);
    ~SafetyBeltAlgo();

    bool load_models(const std::vector<ModelConfig> &models);

    void async_infer(const int64_t image_id, const cv::Mat &image, InferCallback infer_callback);
    bool sync_infer(const int64_t image_id, const cv::Mat &image, std::vector<AlgoObject> &objects);

protected:
    std::vector<AlgoObject> filter_infer_result(const gddeploy::InferResult &infer_result,
                                                const std::set<std::string> &labels);

private:
    SafetyBeltAlgoConfig config_;

    class SafetyBeltAlgoPrivate;
    std::unique_ptr<SafetyBeltAlgoPrivate> private_;
};

}// namespace gddi
