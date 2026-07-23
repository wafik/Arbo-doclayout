// Algorithm ported from ppu-doclayout BaseDocLayoutService (MIT).
#include "arboDocLayout/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace arbo::doclayout {

namespace {

float areaOf(const LayoutBox& b) {
    return std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
}

std::vector<LayoutBox> mergeSameLabelContained(
    const std::vector<LayoutBox>& boxes,
    float iominThreshold = 0.6f
) {
    std::set<size_t> suppressed;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed.count(i)) continue;
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed.count(j)) continue;
            if (boxes[i].label != boxes[j].label) continue;
            if (computeIoMin(boxes[i], boxes[j]) > iominThreshold) {
                const float areaI = areaOf(boxes[i]);
                const float areaJ = areaOf(boxes[j]);
                suppressed.insert(areaI >= areaJ ? j : i);
            }
        }
    }
    std::vector<LayoutBox> kept;
    kept.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!suppressed.count(i)) kept.push_back(boxes[i]);
    }
    return kept;
}

std::vector<LayoutBox> applyCrossClassNMS(
    const std::vector<LayoutBox>& boxes,
    float iouThreshold = 0.8f
) {
    std::vector<LayoutBox> kept;
    std::set<size_t> suppressed;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed.count(i)) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed.count(j)) continue;
            if (computeIoU(boxes[i], boxes[j]) > iouThreshold) {
                suppressed.insert(j);
            }
        }
    }
    return kept;
}

} // namespace

float computeIoU(const LayoutBox& a, const LayoutBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    if (intersection == 0.f) return 0.f;
    const float areaA = areaOf(a);
    const float areaB = areaOf(b);
    return intersection / (areaA + areaB - intersection);
}

float computeIoMin(const LayoutBox& a, const LayoutBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    if (intersection == 0.f) return 0.f;
    const float areaA = areaOf(a);
    const float areaB = areaOf(b);
    return intersection / std::min(areaA, areaB);
}

std::vector<LayoutBox> postprocessBoxes(
    const float* data,
    int numBoxes,
    int numCols,
    float threshold
) {
    if (!data || numBoxes <= 0 || numCols < 7) return {};

    struct Scored {
        LayoutBox box;
        float readingOrder = 0.f;
    };
    std::vector<Scored> scored;
    scored.reserve(static_cast<size_t>(numBoxes));

    for (int i = 0; i < numBoxes; ++i) {
        const int offset = i * numCols;
        const float clsId = data[offset];
        const float score = data[offset + 1];
        if (score < threshold) continue;

        const int labelIndex = static_cast<int>(std::lround(clsId));
        LayoutBox box;
        box.labelIndex = labelIndex;
        box.label = labelForIndex(labelIndex);
        box.score = std::round(score * 10000.f) / 10000.f;
        box.x1 = std::round(data[offset + 2] * 10.f) / 10.f;
        box.y1 = std::round(data[offset + 3] * 10.f) / 10.f;
        box.x2 = std::round(data[offset + 4] * 10.f) / 10.f;
        box.y2 = std::round(data[offset + 5] * 10.f) / 10.f;
        box.sourceIndex = i;
        scored.push_back({std::move(box), data[offset + 6]});
    }

    std::sort(scored.begin(), scored.end(),
              [](const Scored& a, const Scored& b) {
                  return a.readingOrder < b.readingOrder;
              });

    std::vector<LayoutBox> boxes;
    boxes.reserve(scored.size());
    for (auto& s : scored) boxes.push_back(std::move(s.box));

    boxes = mergeSameLabelContained(boxes);
    return applyCrossClassNMS(boxes);
}

std::vector<std::vector<int32_t>> extractMasks(
    const int32_t* masksData,
    int numRawBoxes,
    const std::vector<LayoutBox>& boxes
) {
    std::vector<std::vector<int32_t>> masks;
    if (!masksData || numRawBoxes <= 0) return masks;
    masks.reserve(boxes.size());
    for (const auto& box : boxes) {
        if (box.sourceIndex < 0 || box.sourceIndex >= numRawBoxes) {
            masks.emplace_back();
            continue;
        }
        const int32_t* start = masksData + static_cast<size_t>(box.sourceIndex) * kMaskSize;
        masks.emplace_back(start, start + kMaskSize);
    }
    return masks;
}

} // namespace arbo::doclayout
