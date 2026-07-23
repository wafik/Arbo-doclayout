// cli/arbo_doclayout_demo.cpp — analyze one document image with arbo-doclayout.
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "arboDocLayout/engine.hpp"
#include "arboDocLayout/logging.hpp"

int main(int argc, char* argv[]) {
    arbo::doclayout::setLogCallback(arbo::doclayout::makeStderrLogger());

    cxxopts::Options opts("arbo_doclayout_demo", "Document layout analysis with arbo-doclayout");
    opts.add_options()
        ("image", "Path to input image", cxxopts::value<std::string>())
        ("model", "Override ONNX model path", cxxopts::value<std::string>()->default_value(""))
        ("model-url", "Model download URL when --model empty",
            cxxopts::value<std::string>()->default_value(""))
        ("cache-dir", "Model cache directory", cxxopts::value<std::string>()->default_value(""))
        ("threshold", "Score threshold", cxxopts::value<float>()->default_value("0.5"))
        ("include-masks", "V3 only: include 200x200 segmentation masks per box",
            cxxopts::value<bool>()->default_value("false"))
        ("cuda", "Request CUDA execution provider", cxxopts::value<bool>()->default_value("false"))
        ("tensorrt", "Request TensorRT execution provider", cxxopts::value<bool>()->default_value("false"))
        ("fp16", "TensorRT FP16 (default true)", cxxopts::value<bool>()->default_value("true"))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help") || !result.count("image")) {
        std::cout << opts.help() << std::endl;
        return result.count("image") ? 0 : 1;
    }

    arbo::doclayout::EngineConfig cfg;
    cfg.modelPath = result["model"].as<std::string>();
    cfg.modelUrl = result["model-url"].as<std::string>();
    cfg.cacheDir = result["cache-dir"].as<std::string>();
    cfg.threshold = result["threshold"].as<float>();
    cfg.includeMasks = result["include-masks"].as<bool>();
    cfg.useCuda = result["cuda"].as<bool>();
    cfg.useTensorrt = result["tensorrt"].as<bool>();
    cfg.useFp16 = result["fp16"].as<bool>();

    arbo::doclayout::Engine engine(cfg);
    std::cout << "Backend: " << engine.backend() << "\n";

    auto page = engine.analyze(result["image"].as<std::string>());
    std::cout << "Image: " << page.image << "\n";
    if (page.boxes.empty()) {
        std::cerr << "No regions found (" << page.elapsedMs << " ms)\n";
        return 1;
    }
    std::cout << "Boxes: " << page.boxes.size() << " (" << page.elapsedMs << " ms)";
    if (!page.masks.empty()) {
        std::cout << " masks=" << page.masks.size();
    }
    std::cout << "\n";
    for (size_t i = 0; i < page.boxes.size(); ++i) {
        const auto& b = page.boxes[i];
        std::cout << "  [" << i << "] " << b.label
                  << " score=" << b.score
                  << " xyxy=[" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2 << "]";
        if (i < page.masks.size() && !page.masks[i].empty()) {
            // count nonzero pixels as a quick mask summary
            size_t nz = 0;
            for (int32_t v : page.masks[i]) if (v != 0) ++nz;
            std::cout << " mask_nz=" << nz << "/" << page.masks[i].size();
        }
        std::cout << "\n";
    }
    return 0;
}
