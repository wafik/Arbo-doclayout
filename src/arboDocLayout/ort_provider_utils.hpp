#pragma once
// Adapted from arboOCR ort_provider_utils (Apache-2.0).

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace arbo::doclayout::detail {

struct TrtShapeProfile {
    std::string minShape;
    std::string optShape;
    std::string maxShape;
};

std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session* session);
std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session* session);

void configureExecutionProviders(
    Ort::SessionOptions& sessionOptions,
    bool useCuda,
    bool useTensorrt,
    const std::string& trtCacheDir,
    const TrtShapeProfile& profile,
    bool useFp16
);

} // namespace arbo::doclayout::detail
