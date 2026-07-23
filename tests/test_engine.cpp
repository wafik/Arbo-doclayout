#include <doctest/doctest.h>

#include "arboDocLayout/engine.hpp"

using namespace arbo::doclayout;

TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.threshold == doctest::Approx(0.5f));
    CHECK(cfg.modelInputSize == 800);
    CHECK(cfg.modelPath.empty());
    CHECK(cfg.useCuda == false);
    CHECK(cfg.useTensorrt == false);
    CHECK(cfg.useFp16 == true);
}
