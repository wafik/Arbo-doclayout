#include <doctest/doctest.h>

#include "arboDocLayout/postprocess.hpp"

using namespace arbo::doclayout;

TEST_CASE("postprocess filters by threshold and sorts by reading order") {
    float data[] = {
        22, 0.9f, 10, 10, 50, 50, 1, 0,   // text, later
        22, 0.1f, 0, 0, 1, 1, 0, 0,       // below thresh
        21, 0.8f, 60, 10, 100, 40, 0, 0,  // table, first
    };
    auto boxes = postprocessBoxes(data, 3, 8, 0.5f);
    REQUIRE(boxes.size() == 2);
    CHECK(boxes[0].label == "table");
    CHECK(boxes[1].label == "text");
    CHECK(boxes[0].score == doctest::Approx(0.8f));
}

TEST_CASE("same-label contained box is suppressed") {
    float data[] = {
        22, 0.9f, 0, 0, 100, 100, 0, 0,
        22, 0.85f, 10, 10, 40, 40, 1, 0,
    };
    auto boxes = postprocessBoxes(data, 2, 8, 0.5f);
    REQUIRE(boxes.size() == 1);
    CHECK(boxes[0].x1 == doctest::Approx(0.f));
}

TEST_CASE("cross-class NMS suppresses high-IoU different labels") {
    float data[] = {
        22, 0.9f, 0, 0, 100, 100, 0, 0,
        21, 0.8f, 1, 1, 99, 99, 1, 0,
    };
    auto boxes = postprocessBoxes(data, 2, 8, 0.5f);
    REQUIRE(boxes.size() == 1);
    CHECK(boxes[0].label == "text");
}

TEST_CASE("computeIoU disjoint is zero") {
    LayoutBox a{"", 0, 1, 0, 0, 10, 10};
    LayoutBox b{"", 0, 1, 20, 20, 30, 30};
    CHECK(computeIoU(a, b) == doctest::Approx(0.f));
}
