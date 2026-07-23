#pragma once
// Image → model tensor helpers for PP-DocLayoutV2 (ported from ppu-doclayout).

#include <vector>

#include <opencv2/core.hpp>

#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

struct PreprocessResult {
    std::vector<float> chw; // length 3 * size * size, RGB /255
    int inputSize = 0;
    int origWidth = 0;
    int origHeight = 0;
    // scale_factor feed: [inputSize/origH, inputSize/origW]
    float scaleH = 0.f;
    float scaleW = 0.f;
};

/// Resize to size×size, BGR→RGB, /255, pack CHW.
/// Empty or unsupported Mat → empty chw (never throws).
PreprocessResult preprocessImage(const cv::Mat& bgr, int inputSize = kDefaultModelInputSize);

} // namespace arbo::doclayout
