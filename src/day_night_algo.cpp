#include "day_night_algo.h"
#include "core/result_def.h"
#include "spdlog/spdlog.h"
#include <api/global_config.h>
#include <common/type_convert.h>
#include <mutex>

namespace gddi {

class DayNightAlgo::DayNightAlgoPrivate {
public:
    std::mutex model_mutex;
    std::vector<ModelConfig> model_configs;
    std::vector<std::unique_ptr<gddeploy::InferAPI>> model_impls;
};

DayNightAlgo::DayNightAlgo(const DayNightAlgoConfig &config) : config_(config) {
    gddeploy::gddeploy_init("");
    private_ = std::make_unique<DayNightAlgoPrivate>();
}

DayNightAlgo::~DayNightAlgo() {
    std::lock_guard<std::mutex> lock(private_->model_mutex);
    for (auto &impl : private_->model_impls) { impl->WaitTaskDone(); }
}

bool DayNightAlgo::load_models(const std::vector<ModelConfig> &models) {
    if (models.size() != 1) {
        spdlog::error("DayNightAlgo only support one model");
        return false;
    }

    std::lock_guard<std::mutex> lock(private_->model_mutex);
    private_->model_impls.clear();

    private_->model_configs = models;
    for (const auto &model : models) {
        auto algo_impl = std::make_unique<gddeploy::InferAPI>();
        if (algo_impl->Init("", model.path, model.license, gddeploy::ENUM_API_TYPE::ENUM_API_SESSION_API) != 0) {
            spdlog::error("Failed to load model: {} - {}", model.name, model.path);
            return false;
        }
        private_->model_impls.emplace_back(std::move(algo_impl));
    }

    return true;
}

void DayNightAlgo::async_infer(const int64_t image_id, const cv::Mat &image, InferCallback infer_callback) {
    gddeploy::BufSurfWrapperPtr surface;
    convertMat2BufSurface(const_cast<cv::Mat &>(image), surface);

    auto package = gddeploy::Package::Create(1);
    package->data[0]->Set(surface);

    private_->model_impls[0]->InferAsync(
        package,
        [this, image_id, image, infer_callback](gddeploy::Status status, gddeploy::PackagePtr data,
                                                gddeploy::any user_data) {
            std::vector<AlgoObject> infer_objects;
            if (!data->data.empty() && data->data[0]->HasMetaValue()) {
                infer_objects = parse_infer_result(data->data[0]->GetMetaData<gddeploy::InferResult>());
            }

            if (infer_callback) { infer_callback(image_id, image, infer_objects); }
        });
}

bool DayNightAlgo::sync_infer(const int64_t image_id, const cv::Mat &image, std::vector<AlgoObject> &infer_objects) {
    gddeploy::BufSurfWrapperPtr surface;
    convertMat2BufSurface(const_cast<cv::Mat &>(image), surface);

    auto in_package = gddeploy::Package::Create(1);
    in_package->data[0]->Set(surface);

    auto out_package = gddeploy::Package::Create(1);
    if (private_->model_impls[0]->InferSync(in_package, out_package) != 0) { return false; }

    if (!out_package->data.empty() && out_package->data[0]->HasMetaValue()) {
        infer_objects = parse_infer_result(out_package->data[0]->GetMetaData<gddeploy::InferResult>());
        infer_objects[0].rect = cv::Rect{0, 0, image.cols, image.rows};
    }

    return true;
}

std::vector<AlgoObject> DayNightAlgo::parse_infer_result(const gddeploy::InferResult &infer_result) {
    std::vector<AlgoObject> objects;

    for (auto result_type : infer_result.result_type) {
        if (result_type == gddeploy::GDD_RESULT_TYPE_CLASSIFY) {
            for (const auto &item : infer_result.classify_result.detect_imgs) {
                objects.emplace_back(AlgoObject{0, item.detect_objs[0].class_id, item.detect_objs[0].label,
                                                item.detect_objs[0].score, cv::Rect{}});
            }
        }
    }

    return objects;
}

}// namespace gddi