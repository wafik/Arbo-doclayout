#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>

#include "arboDocLayout/preprocess.hpp"

using namespace arbo::doclayout;

TEST_CASE("preprocessImage produces CHW length 3*S*S in [0,1]") {
    cv::Mat img(120, 80, CV_8UC3, cv::Scalar(0, 128, 255)); // BGR
    auto r = preprocessImage(img, 64);
    REQUIRE(r.chw.size() == static_cast<size_t>(3 * 64 * 64));
    CHECK(r.inputSize == 64);
    CHECK(r.origWidth == 80);
    CHECK(r.origHeight == 120);
    CHECK(r.scaleH == doctest::Approx(64.f / 120.f));
    CHECK(r.scaleW == doctest::Approx(64.f / 80.f));
    for (float v : r.chw) {
        CHECK(v >= 0.f);
        CHECK(v <= 1.f);
    }
    // B=0 G=128 R=255 → RGB R≈1 at channel 0
    CHECK(r.chw[0] == doctest::Approx(1.f).epsilon(0.02));
}

TEST_CASE("preprocessImage empty Mat returns empty") {
    cv::Mat empty;
    auto r = preprocessImage(empty, 32);
    CHECK(r.chw.empty());
}
