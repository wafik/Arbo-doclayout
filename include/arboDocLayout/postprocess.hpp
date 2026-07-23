#pragma once
// V2 box decode + reading-order sort + same-label IoMin merge + cross-class NMS.
// Algorithm ported from ppu-doclayout BaseDocLayoutService (MIT).

#include <vector>

#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

/// Parse flat model output [numBoxes * numCols].
/// V2: numCols>=7 — [cls, score, x1,y1,x2,y2, reading_order, ...].
/// Filters by threshold, sorts by reading_order, merge IoMin=0.6, NMS IoU=0.8.
std::vector<LayoutBox> postprocessBoxes(
    const float* data,
    int numBoxes,
    int numCols,
    float threshold = kDefaultThreshold
);

float computeIoU(const LayoutBox& a, const LayoutBox& b);
float computeIoMin(const LayoutBox& a, const LayoutBox& b);

} // namespace arbo::doclayout
