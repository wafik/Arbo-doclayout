# arbo-doclayout v1 design

**Date:** 2026-07-23  
**Status:** Approved design  
**Scope:** Standalone C++ PP-DocLayoutV2 library, sibling to arboOCR, mergeable later

## Context

[ppu-doclayout](https://github.com/PT-Perkasa-Pilar-Utama/ppu-doclayout) is a TypeScript/Bun library that runs Paddle PP-DocLayout ONNX models for document layout analysis (25 region classes, reading order). arboOCR is a C++17 OCR stack (OpenCV + ONNXRuntime, `Engine`/`EngineConfig`, doctest, CMake/vcpkg). This project ports the layout core to C++ in a **new sibling repo** at `D:\kerjaan\kreasi\kimfu\cpp\arbo-doclayout` so it can later be linked into arboOCR (or a monorepo) without living inside arboOCR for v1.

Reference algorithm source: `ppu-doclayout/src/core/base-doclayout.service.ts` (preprocess, postprocess, same-label merge, cross-class NMS).

## Goals

1. Ship a usable C++ library: load PP-DocLayoutV2 ONNX, analyze a page image, return labeled boxes in reading order.
2. Match arboOCR conventions: C++17, `Engine` + `EngineConfig`, never-throw analyze, OpenCV `cv::Mat`, ORT CPU/CUDA/TensorRT auto-detect, doctest, CMake presets.
3. Auto-fetch default V2 model when no path is given (ppu-doclayout behavior), with local cache.
4. Stay independent of arboOCR at build time (sibling static lib).

## Non-goals (v1)

- PP-DocLayoutV3 and segmentation masks.
- Browser / WASM / Node bindings.
- OCR text recognition (that is arboOCR).
- Reading-order re-ranking beyond the model’s baked-in order index.
- Tunable IoU/IoMin thresholds as public config (fixed at ppu defaults: IoMin 0.6, cross-class IoU 0.8).
- Live dependency on arboOCR headers or sources.

## Design

### 1. Repo layout

```
arbo-doclayout/
├── include/arboDocLayout/
│   ├── engine.hpp
│   ├── types.hpp
│   ├── preprocess.hpp
│   ├── postprocess.hpp
│   ├── model_loader.hpp
│   └── logging.hpp
├── src/arboDocLayout/
│   ├── engine.cpp
│   ├── types.cpp          # optional: toJson only if needed
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── model_loader.cpp
│   ├── logging.cpp
│   └── ort_provider_utils.cpp
├── cli/arbo_doclayout_demo.cpp
├── tests/
│   ├── test_main.cpp
│   ├── test_preprocess.cpp
│   ├── test_postprocess.cpp
│   ├── test_engine.cpp
│   └── test_model_loader.cpp
├── docs/superpowers/specs/
├── CMakeLists.txt
├── CMakePresets.json      # windows-x64 (+ linux-x64 later if needed)
├── README.md
└── LICENSE                # Apache-2.0 preferred for arboOCR merge; note ppu MIT
```

Namespace: `arbo::doclayout`.

### 2. Public types

```cpp
namespace arbo::doclayout {

// 25 labels — same order as ppu-doclayout LABELS
const std::vector<std::string>& labels(); // or constexpr array + size

struct LayoutBox {
    std::string label;
    int labelIndex = -1;
    float score = 0.f;
    // axis-aligned [x1, y1, x2, y2] in original image pixels
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct PageLayout {
    std::string image;                 // filename or empty for Mat API
    std::vector<LayoutBox> boxes;      // reading order after postprocess
    float elapsedMs = 0.f;
};

struct EngineConfig {
    std::string modelPath;             // empty → auto-fetch default V2
    std::string modelUrl;              // empty → built-in default V2 media URL
    std::string cacheDir;              // empty → platform default cache
    float threshold = 0.5f;
    int modelInputSize = 800;
    bool useCuda = false;
    bool useTensorrt = false;
    bool useFp16 = true;               // TensorRT only
    std::string trtCacheDir = "models/trt_engines";
};

class Engine {
public:
    explicit Engine(const EngineConfig& config);
    std::string backend() const;       // "tensorrt" | "cuda" | "cpu"
    PageLayout analyze(const std::string& imagePath);
    PageLayout analyze(const cv::Mat& image); // expects BGR 8U; empty → empty boxes
};

} // namespace arbo::doclayout
```

### 3. Inference pipeline

1. Decode / accept `cv::Mat` (BGR, `CV_8U`, 1 or 3 channel; gray promoted to BGR).
2. **Preprocess:** resize to `modelInputSize × modelInputSize` (default 800), convert BGR→RGB, scale `/255`, pack CHW `float32` tensor `[1,3,H,W]`.
3. **Feeds** (Paddle export names, matching ppu):
   - `image` — CHW float32
   - `im_shape` — `[1,2]` float32 = `{inputSize, inputSize}`
   - `scale_factor` — `[1,2]` float32 = `{inputSize/origH, inputSize/origW}`
4. **Run** ORT session.
5. **Postprocess V2 only:** each row `numCols` (expect 8 for V2):  
   `[cls_id, score, x1, y1, x2, y2, reading_order, …]`  
   - Drop rows with `score < threshold`  
   - Map `cls_id` → label via fixed 25-label table  
   - Sort by `reading_order`  
   - Same-label contained merge (IoMin > 0.6 → drop smaller)  
   - Cross-class NMS (IoU > 0.8 → keep first in sorted order)  
6. Coordinates stay in **original image space** (model already scales via `scale_factor` the same way ppu does — do not double-scale).

If output `numCols` is not the V2 layout (e.g. 7 for V3), v1 logs a warning and still parses the first 7 columns as V2-compatible fields (cls…order) **or** returns empty boxes if dims are unusable. No mask path.

### 4. Model loading & auto-fetch

- If `modelPath` is non-empty: load that ONNX file (must exist; else construction fails with throw, same class of error as arboOCR missing model at load).
- If `modelPath` is empty:
  1. Resolve `modelUrl` (default:  
     `https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/layout/PP-DocLayoutV2.onnx`).
  2. Cache file: `{cacheDir}/{basename(url)}`.  
     Default `cacheDir`: `%USERPROFILE%\.cache\arbo-doclayout` on Windows, `~/.cache/arbo-doclayout` elsewhere (or `models/` under cwd if preferred in implementation — **use home cache** to match ppu).
  3. If cache hit: load from disk.  
     If miss: download via libcurl, write cache, then load.
- Session creation: same provider selection pattern as arboOCR (`useTensorrt` → TRT if available, else CUDA if `useCuda`/`useTensorrt`, else CPU). Expose `backend()`.

Construction **may throw** on unrecoverable load failure (corrupt ONNX, download failure with no cache). `analyze` **never throws**.

### 5. Logging

Optional process-wide callback (copy arboOCR logging pattern: silent by default, `setLogCallback` / `setMinLogLevel`). Demo may use stderr logger.

### 6. CLI

`arbo_doclayout_demo`:

```
--image <path> (required)
--model <path> (optional)
--model-url <url> (optional)
--threshold <float>
--cuda / --tensorrt / --fp16
```

Print backend, elapsed ms, and each box (index, label, score, xyxy).

### 7. Build

- CMake 3.20+, C++17.
- vcpkg: `onnxruntime`, `opencv`, `curl`, `doctest`, `cxxopts`.
- Preset `windows-x64` first (primary dev machine); system-deps/Jetson path optional later.
- Targets: `arboDocLayout` (static), `arbo_doclayout_tests`, `arbo_doclayout_demo`.

### 8. Tests

| Case | File | Expectation |
|------|------|-------------|
| Preprocess shape / finite values | `test_preprocess.cpp` | CHW length `3*S*S`, values in [0,1] |
| Postprocess synthetic V2 rows | `test_postprocess.cpp` | threshold, sort order, IoMin merge, IoU NMS |
| Labels table size 25 | `test_postprocess.cpp` or `test_engine.cpp` | index 0 = abstract, 22 = text, etc. |
| EngineConfig defaults | `test_engine.cpp` | threshold 0.5, input 800, empty modelPath |
| Engine empty Mat | `test_engine.cpp` | empty boxes, no throw (needs model or mock — skip if no model) |
| Model cache path resolution | `test_model_loader.cpp` | path join / basename; download skipped offline |

Inference e2e with real ONNX: skip-by-default in CI if model not present (same pattern as arboOCR `test_engine_inference`).

### 9. Merge path into arboOCR (later, not v1)

1. Publish/install `arboDocLayout` or vendor as subdirectory.
2. Optional `EngineConfig` flag or separate layout stage before OCR detect.
3. Map `LayoutBox` regions (e.g. `table`, `text`) to crop + OCR pipelines.

No shared code required in v1 beyond matching style.

### 10. Licensing note

ppu-doclayout is MIT. Ported algorithm comments should cite ppu-doclayout / Paddle PP-DocLayout. Prefer Apache-2.0 for the C++ repo to align with arboOCR, with `THIRD_PARTY_NOTICES` for upstream.

## Error handling summary

| Stage | Behavior |
|-------|----------|
| Missing/corrupt model at construct | throw |
| Download fail, no cache | throw |
| Empty / unreadable image | empty `boxes`, `elapsedMs` set |
| ORT run exception | log error, empty `boxes` |
| Unsupported Mat depth/channels | empty `boxes` (or convert 1ch→3ch when possible) |

## Alternatives considered

| Approach | Why not |
|----------|---------|
| Implement inside arboOCR immediately | User asked for separate plugin path at `arbo-doclayout` |
| Full ppu V3 + masks | Larger v1; no immediate consumer requirement |
| Path-only load (no auto-fetch) | User chose ppu-style auto-fetch |
| PlatformProvider abstraction | No Web target in C++; dead layers |
| Header-only postprocess lib | Diverges from arboOCR `.hpp`/`.cpp` + doctest style |

## Success criteria

- `Engine` analyzes a sample document image with default V2 model (auto-fetched once) and returns non-empty labeled boxes in reading order when the image has layout structure.
- Unit tests for preprocess/postprocess pass without ONNX.
- CMake `windows-x64` builds library, tests, and demo.
- README documents install, config, CLI, and future arboOCR merge note.
- Zero link dependency on arboOCR.

## Source mapping (ppu → C++)

| ppu-doclayout | arbo-doclayout |
|---------------|----------------|
| `LABELS`, defaults | `types` / constants in headers |
| `BaseDocLayoutService.preprocess` | `preprocess.hpp/.cpp` |
| `postprocess` + IoU/IoMin/NMS | `postprocess.hpp/.cpp` |
| `DocLayoutService.initialize` + fetch | `model_loader` + `Engine` ctor |
| `analyze` | `Engine::analyze` |
| debug draw | optional later; CLI print only in v1 |
