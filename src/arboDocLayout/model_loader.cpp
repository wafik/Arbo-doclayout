#include "arboDocLayout/model_loader.hpp"

#include "arboDocLayout/types.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <curl/curl.h>

namespace fs = std::filesystem;

namespace arbo::doclayout {

namespace {

size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* out = static_cast<std::ofstream*>(userp);
    size_t total = size * nmemb;
    out->write(static_cast<char*>(contents), static_cast<std::streamsize>(total));
    return total;
}

} // namespace

DownloadResult downloadFile(const std::string& url, const std::string& destPath) {
    fs::path dest(destPath);
    if (fs::exists(dest) && fs::file_size(dest) > 0) {
        return {true, "", 0};
    }

    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);
    std::ofstream out(dest, std::ios::binary);
    if (!out.is_open()) {
        return {false, "cannot open destination file for writing: " + destPath, 0};
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        return {false, "curl_easy_init failed", 0};
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "arbo-doclayout/0.1");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    CURLcode res = curl_easy_perform(curl);
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);
    out.close();

    if (res != CURLE_OK || httpCode >= 400) {
        fs::remove(dest, ec);
        std::string err = (res != CURLE_OK)
            ? std::string("curl error: ") + curl_easy_strerror(res)
            : "HTTP " + std::to_string(httpCode);
        return {false, err, 0};
    }

    size_t bytes = fs::exists(dest) ? fs::file_size(dest) : 0;
    return {true, "", bytes};
}

std::string defaultCacheDir() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    if (!home || !*home) {
        return "models";
    }
    return (fs::path(home) / ".cache" / "arbo-doclayout").string();
}

std::string urlFileName(const std::string& url) {
    auto q = url.find('?');
    std::string path = q == std::string::npos ? url : url.substr(0, q);
    auto slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return path;
    std::string name = path.substr(slash + 1);
    return name.empty() ? "model.onnx" : name;
}

std::string resolveModelFile(
    const std::string& modelPath,
    const std::string& modelUrl,
    const std::string& cacheDir
) {
    if (!modelPath.empty()) {
        return modelPath;
    }

    const std::string url = modelUrl.empty() ? std::string(kDefaultModelUrl) : modelUrl;
    const std::string cache = cacheDir.empty() ? defaultCacheDir() : cacheDir;
    const fs::path dest = fs::path(cache) / urlFileName(url);

    if (fs::exists(dest) && fs::file_size(dest) > 0) {
        return dest.string();
    }

    auto dl = downloadFile(url, dest.string());
    if (!dl.ok) {
        throw std::runtime_error("failed to download model: " + dl.errorMessage + " (" + url + ")");
    }
    return dest.string();
}

} // namespace arbo::doclayout
