#include "arboDocLayout/types.hpp"

namespace arbo::doclayout {

std::string labelForIndex(int labelIndex) {
    if (labelIndex >= 0 && labelIndex < kLabelCount) {
        return kLabels[static_cast<size_t>(labelIndex)];
    }
    return "unknown_" + std::to_string(labelIndex);
}

} // namespace arbo::doclayout
