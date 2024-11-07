/**
 * @file raise_dust_algo.h
 * @author kangpeng (kangpeng@glasssix.com)
 * @brief 工地扬尘检测
 * @version 1.0.0
 * @date 2024-11-07
 * 
 * @copyright Copyright (c) 2024 by Glasssix
 * 
 */

#pragma once

#include "struct_def.h"
#include <api/infer_api.h>
#include <core/result_def.h>
#include <set>

namespace gddi {

struct Raise_DustAlgoConfig {
    float statistics_interval{1};   // 每隔N统计一次
    float statistics_threshold{0.1};// 统计阈值(手与香烟重叠时间占比)
};

class Raise_DustAlgo {
public:
    Raise_DustAlgo(const Raise_DustAlgoConfig &config);
    ~Raise_DustAlgo();

    /**
     * @brief 加载模型
     * 
     * @param models
     * @return true 
     * @return false 
     */
    bool load_models(const std::vector<ModelConfig> &models);

    /**
     * @brief 同步推理接口
     * 
     * @param image_id 
     * @param image 
     * @param objects 
     * @return true 
     * @return false 
     */
    bool sync_infer(const int64_t image_id, const cv::Mat &image, std::vector<AlgoObject> &objects);

protected:
    std::vector<AlgoObject> parse_infer_result(const gddeploy::InferResult &infer_result, const float threshold);

private:
    Raise_DustAlgoConfig config_;

    class Raise_DustAlgoPrivate;
    std::unique_ptr<Raise_DustAlgoPrivate> private_;
};

}// namespace gddi