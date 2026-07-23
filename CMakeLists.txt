cmake_minimum_required(VERSION 3.20)
project(arboDocLayout CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

add_library(arboDocLayout STATIC
    src/arboDocLayout/types.cpp
    src/arboDocLayout/logging.cpp
    src/arboDocLayout/preprocess.cpp
    src/arboDocLayout/postprocess.cpp
    src/arboDocLayout/model_loader.cpp
    src/arboDocLayout/ort_provider_utils.cpp
    src/arboDocLayout/engine.cpp
)
target_include_directories(arboDocLayout PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(arboDocLayout PUBLIC ${OpenCV_LIBS} CURL::libcurl onnxruntime::onnxruntime)

enable_testing()
add_executable(arbo_doclayout_tests
    tests/test_main.cpp
    tests/test_types.cpp
    tests/test_preprocess.cpp
    tests/test_postprocess.cpp
    tests/test_model_loader.cpp
    tests/test_engine.cpp
    tests/test_engine_inference.cpp
)
target_link_libraries(arbo_doclayout_tests PRIVATE arboDocLayout doctest::doctest)
add_test(NAME arbo_doclayout_tests COMMAND arbo_doclayout_tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:arbo_doclayout_tests>)

add_executable(arbo_doclayout_demo cli/arbo_doclayout_demo.cpp)
target_link_libraries(arbo_doclayout_demo PRIVATE arboDocLayout cxxopts::cxxopts)
