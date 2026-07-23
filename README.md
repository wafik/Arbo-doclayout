# arbo-doclayout

**Standalone C++ document layout analysis — PP-DocLayoutV2 on CPU, CUDA, or TensorRT.**

Sibling library to [arboOCR](https://github.com/wafik/ArboOCR). Detects 25 document region types (text, table, image, formula, …) in reading order.

## Quickstart

```cpp
#include <arboDocLayout/engine.hpp>
#include <iostream>

int main() {
    arbo::doclayout::EngineConfig cfg;
    // empty modelPath → auto-fetch default PP-DocLayoutV2.onnx into ~/.cache/arbo-doclayout
    arbo::doclayout::Engine engine(cfg);
    std::cout << "Backend: " << engine.backend() << "\n";

    auto page = engine.analyze("page.jpg"); // never throws
    for (auto& b : page.boxes) {
        std::cout << b.label << " " << b.score
                  << " [" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2 << "]\n";
    }
}
```

CLI:

```bash
./arbo_doclayout_demo --image page.jpg
./arbo_doclayout_demo --image page.jpg --model models/PP-DocLayoutV2.onnx
```

## Build (Windows / vcpkg)

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
cmake --preset windows-x64
cmake --build build/windows-x64 --config Release
./build/windows-x64/Release/arbo_doclayout_tests
```

## EngineConfig

| Field | Default | Notes |
|-------|---------|--------|
| `modelPath` | empty | Local ONNX; empty → download |
| `modelUrl` | PP-DocLayoutV2 media URL | Used when path empty |
| `cacheDir` | `~/.cache/arbo-doclayout` | Download cache |
| `threshold` | `0.5` | Min box score |
| `modelInputSize` | `800` | Square model input |
| `includeMasks` | `false` | V3 only — fill `PageLayout::masks` (200×200 int32 per box) |
| `useCuda` / `useTensorrt` | `false` | ORT providers |
| `useFp16` | `true` | TensorRT only |

### V3 masks

PP-DocLayoutV3 outputs per-region masks. Point `modelPath` / `modelUrl` at a V3 ONNX
and set `includeMasks = true`:

```cpp
cfg.modelUrl = arbo::doclayout::kDefaultV3ModelUrl; // or local path
cfg.includeMasks = true;
auto page = engine.analyze("page.jpg");
// page.masks[i] is 200*200 int32, aligned with page.boxes[i]
```

CLI: `--model path/to/PP-DocLayoutV3.onnx --include-masks`

V2 models ignore `includeMasks` (boxes only).

## Labels (25)

`abstract` · `algorithm` · `aside_text` · `chart` · `content` · `display_formula` · `doc_title` · `figure_title` · `footer` · `footer_image` · `footnote` · `formula_number` · `header` · `header_image` · `image` · `inline_formula` · `number` · `paragraph_title` · `reference` · `reference_content` · `seal` · `table` · `text` · `vertical_text` · `vision_footnote`

## Merge into arboOCR (later)

v1 does **not** link arboOCR. Later: install this static lib and run layout before OCR crops, or vendor as a subdirectory.

## Non-goals (still)

Browser/WASM, OCR text recognition, tunable IoU/IoMin public knobs.

## License

Apache-2.0. Layout algorithm ported from [ppu-doclayout](https://github.com/PT-Perkasa-Pilar-Utama/ppu-doclayout) (MIT) / Paddle PP-DocLayout. See `THIRD_PARTY_NOTICES.md`.
