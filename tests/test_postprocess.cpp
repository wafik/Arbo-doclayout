#include <algorithm>
#include <doctest/doctest.h>
#include <vector>

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

TEST_CASE("postprocess sets sourceIndex for mask lookup") {
    float data[] = {
        22, 0.9f, 10, 10, 50, 50, 0,  // row 0
        21, 0.8f, 60, 10, 100, 40, 1, // row 1
    };
    auto boxes = postprocessBoxes(data, 2, 7, 0.5f);
    REQUIRE(boxes.size() == 2);
    CHECK(boxes[0].sourceIndex == 0);
    CHECK(boxes[1].sourceIndex == 1);
}

TEST_CASE("extractMasks slices by sourceIndex after reorder") {
    // Two raw rows; after sort by reading_order box order is row1 then row0
    float data[] = {
        22, 0.9f, 10, 10, 50, 50, 1, // reading order 1 → second
        21, 0.8f, 60, 10, 100, 40, 0, // reading order 0 → first
    };
    auto boxes = postprocessBoxes(data, 2, 7, 0.5f);
    REQUIRE(boxes.size() == 2);
    CHECK(boxes[0].sourceIndex == 1);
    CHECK(boxes[1].sourceIndex == 0);

    std::vector<int32_t> flat(static_cast<size_t>(2 * kMaskSize), 0);
    // fill raw row0 with 1s, row1 with 2s
    std::fill(flat.begin(), flat.begin() + kMaskSize, 1);
    std::fill(flat.begin() + kMaskSize, flat.end(), 2);

    auto masks = extractMasks(flat.data(), 2, boxes);
    REQUIRE(masks.size() == 2);
    REQUIRE(masks[0].size() == static_cast<size_t>(kMaskSize));
    REQUIRE(masks[1].size() == static_cast<size_t>(kMaskSize));
    CHECK(masks[0][0] == 2); // first kept box was sourceIndex 1
    CHECK(masks[1][0] == 1);
}
