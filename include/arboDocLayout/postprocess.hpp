#pragma once
// V2/V3 box decode + reading-order sort + same-label IoMin merge + cross-class NMS.
// V3 masks: extractMasks (200×200 per source row). Ported from ppu-doclayout (MIT).

#include <cstdint>
#include <vector>

#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

/// Parse flat model output [numBoxes * numCols].
/// V2: numCols==8, V3: numCols==7 — [cls, score, x1,y1,x2,y2, reading_order, ...].
/// Filters by threshold, sorts by reading_order, merge IoMin=0.6, NMS IoU=0.8.
/// Sets LayoutBox::sourceIndex to the raw row index for mask lookup.
std::vector<LayoutBox> postprocessBoxes(
    const float* data,
    int numBoxes,
    int numCols,
    float threshold = kDefaultThreshold
);

/// Slice V3 mask tensor into one kMaskSize vector per kept box (by sourceIndex).
/// masksData is row-major [numBoxes * kMaskSize] int32. Empty sourceIndex → empty mask.
std::vector<std::vector<int32_t>> extractMasks(
    const int32_t* masksData,
    int numRawBoxes,
    const std::vector<LayoutBox>& boxes
);

float computeIoU(const LayoutBox& a, const LayoutBox& b);
float computeIoMin(const LayoutBox& a, const LayoutBox& b);

} // namespace arbo::doclayout
