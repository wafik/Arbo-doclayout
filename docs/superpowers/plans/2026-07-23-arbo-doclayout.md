# arbo-doclayout v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone C++17 PP-DocLayoutV2 library (`arbo::doclayout::Engine`) with auto-fetch, OpenCV+ORT backends, doctest suite, and demo CLI — sibling to arboOCR, no arboOCR link.

**Architecture:** Thin port of ppu-doclayout core: pure `preprocess`/`postprocess` units + `model_loader` (curl cache) + `Engine` owning one ORT session. Same CMake/vcpkg shape as arboOCR. V2 boxes only; no masks, no OCR.

**Tech Stack:** C++17, OpenCV, ONNXRuntime, libcurl, doctest, cxxopts, CMake 3.20 + vcpkg (`windows-x64` preset).

**Spec:** `docs/superpowers/specs/2026-07-23-arbo-doclayout-design.md`

**Working directory:** `D:\kerjaan\kreasi\kimfu\cpp\arbo-doclayout` (not the arboOCR worktree).

---

### File map

| Path | Role |
|------|------|
| `include/arboDocLayout/types.hpp` | `LayoutBox`, `PageLayout`, `LABELS` |
| `include/arboDocLayout/preprocess.hpp` | `preprocessImage` → CHW float tensor + scale helpers |
| `include/arboDocLayout/postprocess.hpp` | parse V2 rows, sort, IoMin merge, cross-class NMS |
| `include/arboDocLayout/model_loader.hpp` | resolve path / download+cache |
| `include/arboDocLayout/logging.hpp` | silent-by-default log callback |
| `include/arboDocLayout/engine.hpp` | `EngineConfig`, `Engine` |
| `src/arboDocLayout/*.cpp` | implementations |
| `src/arboDocLayout/ort_provider_utils.hpp/.cpp` | input/output names + EP config (adapted from arboOCR, own namespace) |
| `cli/arbo_doclayout_demo.cpp` | demo CLI |
| `tests/*` | doctest |
| `CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json`, `README.md`, `LICENSE`, `.gitignore` | build + docs |

---

### Task 1: Scaffold — CMake, presets, gitignore, LICENSE, types + logging

**Files:**
- Create: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Create: `vcpkg.json`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `include/arboDocLayout/types.hpp`
- Create: `include/arboDocLayout/logging.hpp`
- Create: `src/arboDocLayout/logging.cpp`
- Create: `src/arboDocLayout/types.cpp`
- Create: `tests/test_main.cpp`
- Create: `tests/test_types.cpp`

- [ ] **Step 1: Write `.gitignore`**

```
build/
.vs/
out/
models/
*.user
CMakeUserPresets.json
.cache/
```

- [ ] **Step 2: Write `vcpkg.json`**

```json
{
  "name": "arbo-doclayout",
  "version-string": "0.1.0",
  "dependencies": [
    "opencv",
    "onnxruntime",
    "curl",
    "doctest",
    "cxxopts"
  ]
}
```

- [ ] **Step 3: Write `CMakePresets.json`**

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "windows-x64",
      "generator": "Visual Studio 17 2022",
      "architecture": "x64",
      "binaryDir": "${sourceDir}/build/windows-x64",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    },
    {
      "name": "linux-x64",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux-x64",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release" }
    }
  ]
}
```

- [ ] **Step 4: Write `include/arboDocLayout/types.hpp`**

```cpp
#pragma once
// Ported labels/order from ppu-doclayout (MIT) — see THIRD_PARTY_NOTICES / README.

#include <array>
#include <string>
#include <vector>

namespace arbo::doclayout {

/// 25 PP-DocLayout region labels (index == model cls_id).
inline constexpr std::array<const char*, 25> kLabels = {
    "abstract",        "algorithm",      "aside_text",       "chart",
    "content",         "display_formula","doc_title",        "figure_title",
    "footer",          "footer_image",   "footnote",         "formula_number",
    "header",          "header_image",   "image",            "inline_formula",
    "number",          "paragraph_title","reference",        "reference_content",
    "seal",            "table",          "text",             "vertical_text",
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
```

- [ ] **Step 5: Write `src/arboDocLayout/types.cpp`**

```cpp
#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

std::string labelForIndex(int labelIndex) {
    if (labelIndex >= 0 && labelIndex < kLabelCount) {
        return kLabels[static_cast<size_t>(labelIndex)];
    }
    return "unknown_" + std::to_string(labelIndex);
}

} // namespace arbo::doclayout
```

- [ ] **Step 6: Write logging header + cpp (arboOCR pattern, rebranded)**

`include/arboDocLayout/logging.hpp`:

```cpp
#pragma once

#include <functional>
#include <string>

namespace arbo::doclayout {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

using LogCallback = std::function<void(LogLevel, const std::string&)>;

void setLogCallback(LogCallback callback);
void setMinLogLevel(LogLevel level);
LogLevel minLogLevel();
void log(LogLevel level, const std::string& message);
LogCallback makeStderrLogger();

} // namespace arbo::doclayout
```

`src/arboDocLayout/logging.cpp`: copy arboOCR `logging.cpp` logic; prefix stderr lines with `[arboDocLayout]` instead of `[arboOCR]`.

- [ ] **Step 7: Write minimal `CMakeLists.txt` (library + tests stubs)**

```cmake
cmake_minimum_required(VERSION 3.20)
project(arboDocLayout CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

add_library(arboDocLayout STATIC
    src/arboDocLayout/types.cpp
    src/arboDocLayout/logging.cpp
    # later tasks add: preprocess, postprocess, model_loader, ort_provider_utils, engine
)
target_include_directories(arboDocLayout PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(arboDocLayout PUBLIC ${OpenCV_LIBS} CURL::libcurl onnxruntime::onnxruntime)

enable_testing()
add_executable(arbo_doclayout_tests
    tests/test_main.cpp
    tests/test_types.cpp
)
target_link_libraries(arbo_doclayout_tests PRIVATE arboDocLayout doctest::doctest)
add_test(NAME arbo_doclayout_tests COMMAND arbo_doclayout_tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:arbo_doclayout_tests>)
```

- [ ] **Step 8: Write `tests/test_main.cpp` and `tests/test_types.cpp`**

```cpp
// tests/test_main.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
```

```cpp
// tests/test_types.cpp
#include <doctest/doctest.h>
#include "arboDocLayout/types.hpp"

using namespace arbo::doclayout;

TEST_CASE("kLabels has 25 entries and known indices") {
    CHECK(kLabelCount == 25);
    CHECK(std::string(kLabels[0]) == "abstract");
    CHECK(std::string(kLabels[21]) == "table");
    CHECK(std::string(kLabels[22]) == "text");
    CHECK(labelForIndex(22) == "text");
    CHECK(labelForIndex(99) == "unknown_99");
}
```

- [ ] **Step 9: LICENSE Apache-2.0**

Copy Apache-2.0 text (same as arboOCR LICENSE) into `LICENSE`.

- [ ] **Step 10: Configure, build, run types tests**

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
cd D:\kerjaan\kreasi\kimfu\cpp\arbo-doclayout
cmake --preset windows-x64
cmake --build build/windows-x64 --config Release --target arbo_doclayout_tests
./build/windows-x64/Release/arbo_doclayout_tests
```

Expected: PASS (1 test case).

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "chore: scaffold arbo-doclayout CMake, types, logging"
```

---

### Task 2: Preprocess unit + tests

**Files:**
- Create: `include/arboDocLayout/preprocess.hpp`
- Create: `src/arboDocLayout/preprocess.cpp`
- Create: `tests/test_preprocess.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write header**

```cpp
#pragma once
// Image → model tensor helpers for PP-DocLayoutV2 (ported from ppu-doclayout).

#include <vector>
#include <opencv2/core.hpp>

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
```

(Need `#include "arboDocLayout/types.hpp"` for `kDefaultModelInputSize` or hardcode default 800 in signature only — use types.hpp.)

- [ ] **Step 2: Failing tests**

```cpp
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
    // Top-left after resize: solid color — R plane should be ~1 (BGR 255,128,0 → RGB 0,128,255)
    // B=0 G=128 R=255 → RGB R=255/255=1 at channel0
    CHECK(r.chw[0] == doctest::Approx(1.f).epsilon(0.02));
}

TEST_CASE("preprocessImage empty Mat returns empty") {
    cv::Mat empty;
    auto r = preprocessImage(empty, 32);
    CHECK(r.chw.empty());
}
```

- [ ] **Step 3: Implement `preprocess.cpp`**

```cpp
#include "arboDocLayout/preprocess.hpp"
#include "arboDocLayout/types.hpp"

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
```

- [ ] **Step 4: Add sources to CMake, build, run**

Add `src/arboDocLayout/preprocess.cpp` to library and `tests/test_preprocess.cpp` to tests.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: preprocessImage CHW tensor for DocLayout"
```

---

### Task 3: Postprocess unit + tests

**Files:**
- Create: `include/arboDocLayout/postprocess.hpp`
- Create: `src/arboDocLayout/postprocess.cpp`
- Create: `tests/test_postprocess.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write header**

```cpp
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
```

- [ ] **Step 2: Failing tests**

```cpp
#include <doctest/doctest.h>
#include "arboDocLayout/postprocess.hpp"

using namespace arbo::doclayout;

TEST_CASE("postprocess filters by threshold and sorts by reading order") {
    // two boxes: high score order=1, low score order=0 (filtered), mid score order=0
    // row: cls, score, x1,y1,x2,y2, order, pad
    float data[] = {
        22, 0.9f, 10, 10, 50, 50, 1, 0,   // text, later in order
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
    // outer and inner same label; IoMin high
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
        22, 0.9f, 0, 0, 100, 100, 0, 0,  // text kept
        21, 0.8f, 1, 1, 99, 99, 1, 0,    // table almost same box → suppressed
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
```

- [ ] **Step 3: Implement postprocess.cpp**

Port logic from ppu `postprocess` / `mergeSameLabelContained` / `applyCrossClassNMS` / `computeIoU` / `computeIoMin`:

- Skip if `numCols < 7` or `data == nullptr` or `numBoxes <= 0` → empty vector.
- Round score to 4 decimals, coords to 1 decimal (match ppu).
- `labelForIndex(round(cls_id))`.

Fixed constants: `iominThreshold = 0.6f`, `iouThreshold = 0.8f` (not configurable in v1).

- [ ] **Step 4: CMake + build + test**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: V2 postprocess with reading order, merge, NMS"
```

---

### Task 4: Model loader (path + auto-fetch cache)

**Files:**
- Create: `include/arboDocLayout/model_loader.hpp`
- Create: `src/arboDocLayout/model_loader.cpp`
- Create: `tests/test_model_loader.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Header**

```cpp
#pragma once

#include <string>

namespace arbo::doclayout {

struct DownloadResult {
    bool ok = false;
    std::string errorMessage;
    size_t bytesWritten = 0;
};

/// Download url → destPath. Skip if dest exists and non-empty. Never throws.
DownloadResult downloadFile(const std::string& url, const std::string& destPath);

/// Default cache directory: %USERPROFILE%\.cache\arbo-doclayout (Win) or ~/.cache/arbo-doclayout.
std::string defaultCacheDir();

/// Basename of URL path (e.g. PP-DocLayoutV2.onnx).
std::string urlFileName(const std::string& url);

/// Resolve ONNX path for Engine:
/// - if modelPath non-empty → return it (caller must ensure exists)
/// - else download modelUrl (or kDefaultModelUrl) into cacheDir (or defaultCacheDir)
/// Throws std::runtime_error if download fails and no cache file.
std::string resolveModelFile(
    const std::string& modelPath,
    const std::string& modelUrl,
    const std::string& cacheDir
);

} // namespace arbo::doclayout
```

- [ ] **Step 2: Tests (no network required)**

```cpp
TEST_CASE("urlFileName extracts basename") {
    CHECK(urlFileName("https://example.com/a/PP-DocLayoutV2.onnx") == "PP-DocLayoutV2.onnx");
}

TEST_CASE("defaultCacheDir is non-empty") {
    CHECK_FALSE(defaultCacheDir().empty());
}

TEST_CASE("resolveModelFile returns explicit path unchanged") {
    CHECK(resolveModelFile("C:/models/x.onnx", "", "") == "C:/models/x.onnx");
}
```

- [ ] **Step 3: Implement** (curl write callback like arboOCR `model_downloader.cpp`; User-Agent `arbo-doclayout/0.1`)

Windows home: `std::getenv("USERPROFILE")` then `/.cache/arbo-doclayout`.  
Else `HOME` + `/.cache/arbo-doclayout`.

- [ ] **Step 4: Build + test + commit**

```bash
git commit -am "feat: model path resolve and curl auto-fetch cache"
```

---

### Task 5: ORT provider utils + Engine

**Files:**
- Create: `include/arboDocLayout/engine.hpp`
- Create: `src/arboDocLayout/engine.cpp`
- Create: `src/arboDocLayout/ort_provider_utils.hpp`
- Create: `src/arboDocLayout/ort_provider_utils.cpp`
- Create: `tests/test_engine.cpp`
- Create: `tests/test_engine_inference.cpp` (skipped by default)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: `ort_provider_utils`**

Copy arboOCR `ort_provider_utils` into `arbo::doclayout::detail` namespace. For DocLayout TRT profiles use fixed shapes matching feeds:

```
image: min/opt/max = 1x3x800x800 (or use config input size — v1 fixed 800 for TRT profile)
im_shape: 1x2
scale_factor: 1x2
```

If TRT shape strings are awkward for multi-input, v1 may enable CUDA-only path when TensorRT fails — prefer: set profile for `image` only if ORT allows; otherwise document CPU/CUDA first and still attempt TRT with image profile.

Practical v1: reuse configure pattern; profile:

```cpp
TrtShapeProfile{
  "image:1x3x800x800,im_shape:1x2,scale_factor:1x2",
  "image:1x3x800x800,im_shape:1x2,scale_factor:1x2",
  "image:1x3x800x800,im_shape:1x2,scale_factor:1x2",
};
```

(Adjust if ORT expects different format — match arboOCR detector profile style.)

- [ ] **Step 2: `engine.hpp`**

```cpp
#pragma once

#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

struct EngineConfig {
    std::string modelPath;
    std::string modelUrl;   // empty → kDefaultModelUrl
    std::string cacheDir;   // empty → defaultCacheDir()
    float threshold = kDefaultThreshold;
    int modelInputSize = kDefaultModelInputSize;
    bool useCuda = false;
    bool useTensorrt = false;
    bool useFp16 = true;
    std::string trtCacheDir = "models/trt_engines";
};

class Engine {
public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    std::string backend() const { return backend_; }

    /// Never throws.
    PageLayout analyze(const std::string& imagePath);
    PageLayout analyze(const cv::Mat& image);

private:
    PageLayout run(const cv::Mat& src, const std::string& imageName);

    EngineConfig config_;
    std::string backend_ = "cpu";
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "arboDocLayout"};
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;
};

bool detectCuda();
bool detectTensorrt();

} // namespace arbo::doclayout
```

- [ ] **Step 3: Engine constructor**

1. `path = resolveModelFile(config.modelPath, config.modelUrl, config.cacheDir)` — may throw.
2. Detect providers; set `backend_`.
3. `configureExecutionProviders(...)`.
4. Create `Ort::Session` from path (wide string on Windows — match arboOCR detector if it uses `std::wstring` conversion).

Check arboOCR detector loadModel for Windows path handling and mirror it.

- [ ] **Step 4: `run` pipeline**

```cpp
PageLayout Engine::run(const cv::Mat& src, const std::string& imageName) {
    auto t0 = steady_clock::now();
    PageLayout result;
    result.image = imageName;
    if (src.empty()) {
        result.elapsedMs = ...;
        return result;
    }
    try {
        auto prep = preprocessImage(src, config_.modelInputSize);
        if (prep.chw.empty()) { ... return; }

        // Create tensors image, im_shape, scale_factor
        // Feed by name: prefer names from session inputNames; fallback "image","im_shape","scale_factor"
        auto outputs = session_->Run(...);
        // first output: float data, dims [N, C]
        result.boxes = postprocessBoxes(data, n, c, config_.threshold);
    } catch (const std::exception& ex) {
        log(LogLevel::Error, std::string("analyze failed: ") + ex.what());
        result.boxes.clear();
    }
    result.elapsedMs = ...;
    return result;
}
```

`analyze(path)`: `cv::imread` then `run`.

- [ ] **Step 5: Default-value tests**

```cpp
TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.threshold == doctest::Approx(0.5f));
    CHECK(cfg.modelInputSize == 800);
    CHECK(cfg.modelPath.empty());
    CHECK(cfg.useCuda == false);
    CHECK(cfg.useTensorrt == false);
    CHECK(cfg.useFp16 == true);
}
```

- [ ] **Step 6: Inference test skipped**

```cpp
TEST_CASE("Engine analyze sample" * doctest::skip(true)) {
    EngineConfig cfg;
    // optional: cfg.modelPath = "models/PP-DocLayoutV2.onnx";
    Engine engine(cfg);
    auto page = engine.analyze("tests/fixtures/sample.png");
    CHECK(page.elapsedMs >= 0.f);
}
```

- [ ] **Step 7: Build full tests + commit**

```bash
git commit -am "feat: Engine load + analyze PP-DocLayoutV2"
```

---

### Task 6: CLI demo

**Files:**
- Create: `cli/arbo_doclayout_demo.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Demo source**

Mirror `arboocr_demo.cpp`:

```cpp
// flags: image (required), model, model-url, cache-dir, threshold, cuda, tensorrt, fp16, help
// setLogCallback(makeStderrLogger());
// print Backend, Image, Boxes count, each [i] label score xyxy
```

- [ ] **Step 2: CMake executable `arbo_doclayout_demo`** linked to `arboDocLayout` + `cxxopts::cxxopts`.

- [ ] **Step 3: Build and `--help`**

Expected: help lists `--image`, `--model`, etc.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(cli): arbo_doclayout_demo"
```

---

### Task 7: README + THIRD_PARTY_NOTICES

**Files:**
- Create: `README.md`
- Create: `THIRD_PARTY_NOTICES.md`

- [ ] **Step 1: README**

Sections: what / quickstart C++ / build windows-x64 / models auto-fetch / API EngineConfig / CLI / labels list / merge note for arboOCR / license.

- [ ] **Step 2: THIRD_PARTY_NOTICES**

Credit ppu-doclayout (MIT), Paddle PP-DocLayout, ONNXRuntime, OpenCV, etc.

- [ ] **Step 3: Commit**

```bash
git commit -am "docs: README and third-party notices"
```

---

### Task 8: Final verification

- [ ] **Step 1: Clean rebuild**

```powershell
cmake --build build/windows-x64 --config Release
```

- [ ] **Step 2: Full test suite**

```powershell
./build/windows-x64/Release/arbo_doclayout_tests
```

Expected: all non-skipped green.

- [ ] **Step 3: Optional local e2e** (if network OK)

```powershell
./build/windows-x64/Release/arbo_doclayout_demo --image path\to\doc.jpg
```

First run downloads V2 model to cache.

- [ ] **Step 4: Confirm file set matches plan; no arboOCR dependency in CMake.**

---

### Self-review (plan vs spec)

| Spec item | Task |
|-----------|------|
| V2-only Engine | Task 5 |
| Auto-fetch | Task 4 |
| preprocess/postprocess | Tasks 2–3 |
| CPU/CUDA/TRT | Task 5 |
| never-throw analyze | Task 5 |
| CLI | Task 6 |
| doctest | Tasks 1–5 |
| README | Task 7 |
| No arboOCR link | Task 1 CMake |
| Sibling layout include/arboDocLayout | Task 1 |

No TBD placeholders. Types consistent: `LayoutBox` fields `x1..y2`, `EngineConfig` names match design.

---

### Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-23-arbo-doclayout.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, spec + quality review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
