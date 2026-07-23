#pragma once
// Ported labels/order from ppu-doclayout (MIT) — see THIRD_PARTY_NOTICES / README.

#include <array>
#include <string>
#include <vector>

namespace arbo::doclayout {

/// 25 PP-DocLayout region labels (index == model cls_id).
inline constexpr std::array<const char*, 25> kLabels = {
    "abstract",         "algorithm",       "aside_text",        "chart",
    "content",          "display_formula", "doc_title",         "figure_title",
    "footer",           "footer_image",    "footnote",          "formula_number",
    "header",           "header_image",    "image",             "inline_formula",
    "number",           "paragraph_title", "reference",         "reference_content",
    "seal",             "table",           "text",              "vertical_text",
    "vision_footnote",
};

inline constexpr int kLabelCount = 25;
inline constexpr int kDefaultModelInputSize = 800;
inline constexpr float kDefaultThreshold = 0.5f;
/// V3 per-region binary mask side length (200×200).
inline constexpr int kMaskSide = 200;
inline constexpr int kMaskSize = kMaskSide * kMaskSide;

/// Default V2 ONNX URL (ppu-paddle-ocr-models media).
inline constexpr const char* kDefaultModelUrl =
    "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/"
    "ppu-paddle-ocr-models/main/layout/PP-DocLayoutV2.onnx";

/// Optional V3 ONNX URL (masks available when includeMasks + V3 model).
inline constexpr const char* kDefaultV3ModelUrl =
    "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/"
    "ppu-paddle-ocr-models/main/layout/PP-DocLayoutV3.onnx";

struct LayoutBox {
    std::string label;
    int labelIndex = -1;
    float score = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    /// Index into raw model output rows (before threshold/NMS). Used for V3 masks.
    int sourceIndex = -1;
};

struct PageLayout {
    std::string image;
    std::vector<LayoutBox> boxes;
    /// Per-box 200×200 masks when EngineConfig::includeMasks and V3 model; else empty.
    /// masks[i] pairs with boxes[i] (same length when present).
    std::vector<std::vector<int32_t>> masks;
    float elapsedMs = 0.f;
};

/// Resolve label string for cls index; unknown → "unknown_<n>".
std::string labelForIndex(int labelIndex);

} // namespace arbo::doclayout
