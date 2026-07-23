#include "arboDocLayout/preprocess.hpp"

#include <opencv2/imgproc.hpp>

namespace arbo::doclayout {

PreprocessResult preprocessImage(const cv::Mat& bgr, int inputSize) {
    PreprocessResult out;
    out.inputSize = inputSize;
    if (bgr.empty() || inputSize <= 0) return out;
    if (bgr.depth() != CV_8U) return out;

    cv::Mat src = bgr;
    if (src.channels() == 1) {
        cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
    } else if (src.channels() != 3) {
        return out;
    }

    out.origWidth = src.cols;
    out.origHeight = src.rows;
    out.scaleH = static_cast<float>(inputSize) / static_cast<float>(src.rows);
    out.scaleW = static_cast<float>(inputSize) / static_cast<float>(src.cols);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(inputSize, inputSize), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    const int plane = inputSize * inputSize;
    out.chw.resize(static_cast<size_t>(3 * plane));
    for (int y = 0; y < inputSize; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputSize; ++x) {
            const int i = y * inputSize + x;
            out.chw[static_cast<size_t>(i)] = row[x][0] / 255.f;             // R
            out.chw[static_cast<size_t>(plane + i)] = row[x][1] / 255.f;     // G
            out.chw[static_cast<size_t>(2 * plane + i)] = row[x][2] / 255.f; // B
        }
    }
    return out;
}

} // namespace arbo::doclayout
