#include <doctest/doctest.h>

#include "arboDocLayout/engine.hpp"

using namespace arbo::doclayout;

TEST_CASE("Engine analyze sample" * doctest::skip(true)) {
    EngineConfig cfg;
    // cfg.modelPath = "models/PP-DocLayoutV2.onnx";
    Engine engine(cfg);
    auto page = engine.analyze("tests/fixtures/sample.png");
    CHECK(page.elapsedMs >= 0.f);
}
