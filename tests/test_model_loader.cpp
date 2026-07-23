#include <doctest/doctest.h>

#include "arboDocLayout/model_loader.hpp"

using namespace arbo::doclayout;

TEST_CASE("urlFileName extracts basename") {
    CHECK(urlFileName("https://example.com/a/PP-DocLayoutV2.onnx") == "PP-DocLayoutV2.onnx");
}

TEST_CASE("defaultCacheDir is non-empty") {
    CHECK_FALSE(defaultCacheDir().empty());
}

TEST_CASE("resolveModelFile returns explicit path unchanged") {
    CHECK(resolveModelFile("C:/models/x.onnx", "", "") == "C:/models/x.onnx");
}
