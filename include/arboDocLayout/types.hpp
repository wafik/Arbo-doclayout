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

/// Default V2 ONNX URL (ppu-paddle-ocr-models media).
inline constexpr const char* kDefaultModelUrl =
    "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/"
    "ppu-paddle-ocr-models/main/layout/PP-DocLayoutV2.onnx";

struct LayoutBox {
    std::string label;
    int labelIndex = -1;
    float score = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
};

struct PageLayout {
    std::string image;
    std::vector<LayoutBox> boxes;
    float elapsedMs = 0.f;
};

/// Resolve label string for cls index; unknown → "unknown_<n>".
std::string labelForIndex(int labelIndex);

} // namespace arbo::doclayout
