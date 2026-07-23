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
