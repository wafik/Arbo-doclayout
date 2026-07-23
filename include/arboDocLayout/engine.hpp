#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

struct EngineConfig {
    std::string modelPath;
    std::string modelUrl;   // empty → kDefaultModelUrl
    std::string cacheDir;   // empty → defaultCacheDir()
    float threshold = kDefaultThreshold;
    int modelInputSize = kDefaultModelInputSize;
    bool useCuda = false;
    bool useTensorrt = false;
    bool useFp16 = true;
    std::string trtCacheDir = "models/trt_engines";
};

class Engine {
public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    std::string backend() const { return backend_; }

    /// Never throws.
    PageLayout analyze(const std::string& imagePath);
    PageLayout analyze(const cv::Mat& image);

private:
    PageLayout run(const cv::Mat& src, const std::string& imageName);

    EngineConfig config_;
    std::string backend_ = "cpu";
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "arboDocLayout"};
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;
};

bool detectCuda();
bool detectTensorrt();

} // namespace arbo::doclayout
