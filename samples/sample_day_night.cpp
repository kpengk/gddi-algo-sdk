#include "day_night_algo.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main() {
    gddi::DayNightAlgoConfig config;
    auto day_night_algo = std::make_unique<gddi::DayNightAlgo>(config);

    std::string video_path = "../videos/day_night_1.mp4";
    std::vector<gddi::ModelConfig> models = {
        {"day_night", "../models/day_night.gdd", "../models/license_day_night.gdd", 0.3}};

    if (!day_night_algo->load_models(models)) {
        printf("Failed to load models\n");
        return -1;
    }

    // 读取视频, 进行推理
    auto image = cv::VideoCapture(video_path);
    if (!image.isOpened()) {
        printf("Failed to open video: %s\n", video_path.c_str());
        return -1;
    }

    int64_t frame_index = 0;
    while (true) {
        cv::Mat frame;
        image.read(frame);
        if (frame.empty()) { break; }

        auto start = std::chrono::steady_clock::now();

        std::vector<gddi::AlgoObject> objects;
        day_night_algo->sync_infer(frame_index, frame, objects);

        if (!objects.empty()) {
            printf("=============== Frame: %ld, Objects: %s\n", frame_index, objects[0].label.c_str());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(
            40
            - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()));

        frame_index++;
    }

    printf("Finished\n");

    return 0;
}