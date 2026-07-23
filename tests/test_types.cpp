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
