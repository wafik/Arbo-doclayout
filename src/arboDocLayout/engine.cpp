#include "arboDocLayout/engine.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "arboDocLayout/logging.hpp"
#include "arboDocLayout/model_loader.hpp"
#include "arboDocLayout/postprocess.hpp"
#include "arboDocLayout/preprocess.hpp"
#include "ort_provider_utils.hpp"

namespace fs = std::filesystem;

namespace arbo::doclayout {

bool detectCuda() {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (auto& p : providers) {
            if (p == "CUDAExecutionProvider") return true;
        }
    } catch (...) {
    }
    return false;
}

bool detectTensorrt() {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (auto& p : providers) {
            if (p == "TensorrtExecutionProvider") return true;
        }
    } catch (...) {
    }
    return false;
}

Engine::Engine(const EngineConfig& config) : config_(config) {
    const std::string path = resolveModelFile(config.modelPath, config.modelUrl, config.cacheDir);

    bool useTensorrt = config.useTensorrt && detectTensorrt();
    bool useCuda = (config.useCuda || useTensorrt) && detectCuda();
    backend_ = useTensorrt ? "tensorrt" : useCuda ? "cuda" : "cpu";
    log(LogLevel::Info, "Engine backend: " + backend_
        + (useTensorrt ? (config.useFp16 ? " (fp16)" : " (fp32)") : ""));
    log(LogLevel::Debug, "Model path: " + path);

    sessionOptions_.SetInterOpNumThreads(0);
    sessionOptions_.SetIntraOpNumThreads(0);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const int s = config.modelInputSize > 0 ? config.modelInputSize : kDefaultModelInputSize;
    const std::string shape =
        "image:1x3x" + std::to_string(s) + "x" + std::to_string(s)
        + ",im_shape:1x2,scale_factor:1x2";
    detail::configureExecutionProviders(
        sessionOptions_, useCuda, useTensorrt, config.trtCacheDir,
        {shape, shape, shape}, config.useFp16);

#ifdef _WIN32
    std::wstring wpath(path.begin(), path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), sessionOptions_);
#endif
    inputNamesPtr_ = detail::getInputNames(session_.get());
    outputNamesPtr_ = detail::getOutputNames(session_.get());
}

Engine::~Engine() = default;

PageLayout Engine::run(const cv::Mat& src, const std::string& imageName) {
    auto t0 = std::chrono::steady_clock::now();
    PageLayout result;
    result.image = imageName;

    auto elapsed = [&]() {
        return std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    };

    if (src.empty()) {
        log(LogLevel::Warn, imageName.empty()
            ? "analyze: empty image buffer"
            : "analyze: empty/unreadable image (" + imageName + ")");
        result.elapsedMs = elapsed();
        return result;
    }

    try {
        if (!session_) {
            log(LogLevel::Error, "analyze: session not loaded");
            result.elapsedMs = elapsed();
            return result;
        }

        auto prep = preprocessImage(src, config_.modelInputSize);
        if (prep.chw.empty()) {
            log(LogLevel::Warn, "analyze: preprocess produced empty tensor");
            result.elapsedMs = elapsed();
            return result;
        }

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::array<int64_t, 4> imageShape{1, 3, prep.inputSize, prep.inputSize};
        Ort::Value imageTensor = Ort::Value::CreateTensor<float>(
            memInfo, prep.chw.data(), prep.chw.size(),
            imageShape.data(), imageShape.size());

        float imShapeData[2] = {
            static_cast<float>(prep.inputSize),
            static_cast<float>(prep.inputSize)};
        std::array<int64_t, 2> pairDims{1, 2};
        Ort::Value imShapeTensor = Ort::Value::CreateTensor<float>(
            memInfo, imShapeData, 2, pairDims.data(), pairDims.size());

        float scaleData[2] = {prep.scaleH, prep.scaleW};
        Ort::Value scaleTensor = Ort::Value::CreateTensor<float>(
            memInfo, scaleData, 2, pairDims.data(), pairDims.size());

        // Fixed feed order matching Paddle export / ppu-doclayout.
        const char* inputNames[] = {"image", "im_shape", "scale_factor"};
        Ort::Value inputs[] = {
            std::move(imageTensor),
            std::move(imShapeTensor),
            std::move(scaleTensor),
        };

        std::vector<const char*> outputNames;
        outputNames.reserve(outputNamesPtr_.size());
        for (auto& n : outputNamesPtr_) outputNames.push_back(n.get());
        if (outputNames.empty()) {
            throw std::runtime_error("model has no outputs");
        }

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            inputs,
            3,
            outputNames.data(),
            outputNames.size());

        if (!outputs.empty()) {
            auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() >= 2 && dims[0] > 0 && dims[1] > 0) {
                result.boxes = postprocessBoxes(
                    outputs[0].GetTensorData<float>(),
                    static_cast<int>(dims[0]),
                    static_cast<int>(dims[1]),
                    config_.threshold);
            }
        }

        log(LogLevel::Debug, "analyze: " + std::to_string(result.boxes.size()) + " boxes");
    } catch (const std::exception& ex) {
        log(LogLevel::Error, std::string("analyze failed: ") + ex.what());
        result.boxes.clear();
    }

    result.elapsedMs = elapsed();
    return result;
}

PageLayout Engine::analyze(const std::string& imagePath) {
    fs::path path(imagePath);
    cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (src.empty()) {
        log(LogLevel::Warn, "analyze: failed to read image path: " + imagePath);
    }
    return run(src, path.filename().string());
}

PageLayout Engine::analyze(const cv::Mat& image) {
    return run(image, {});
}

} // namespace arbo::doclayout
