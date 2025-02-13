cmake_minimum_required(VERSION 3.5)

project(wav2wav)
option(BUILD_ONNX "Build on onnx runtime." OFF)
option(BUILD_NNCASE "Build on nncase." ON)

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (BUILD_ONNX)
    add_definitions(-DONNX)
endif()

if (BUILD_ONNX)
    set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)
    include_directories(${ONNXRUNTIME_PATH}/include)
    link_directories(${ONNXRUNTIME_PATH}/lib)
else()
    if(CMAKE_CROSSCOMPILING)
        set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/riscv64)
        link_directories(${CMAKE_SOURCE_DIR}/3rd_party/mmz/riscv64)
    else()
        set(NNCASE_PATH ${CMAKE_SOURCE_DIR}/3rd_party/nncase/x86_64)
    endif()
    include_directories(3rd_party
                        ${NNCASE_PATH}/include
                        ${NNCASE_PATH}/include/nncase/runtime
                        )
    link_directories(${NNCASE_PATH}/lib)
endif()

# librosa
set(ROSA_PATH ${CMAKE_SOURCE_DIR}/3rd_party/librosa)
set(FFTW_PATH ${CMAKE_SOURCE_DIR}/3rd_party/fftw)

set(TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/3rd_party/tokenizers-cpp)

if (CMAKE_CROSSCOMPILING)
link_directories(${TOKENIZER_PATH}/lib/riscv64 ${FFTW_PATH}/lib)
else()
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sndfile/lib/cmake)
    set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})
    find_package(SndFile REQUIRED)

    # tokenizers-cpp
    link_directories(${TOKENIZER_PATH}/lib/x86_64)

    include_directories(${ONNXRUNTIME_PATH}/include)

endif()

include_directories(${ROSA_PATH} ${TOKENIZER_PATH}/include ${CMAKE_SOURCE_DIR}/include)
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_library(wav2wav STATIC ${SRCS}
        src/audio.cpp
        src/utils.cpp)

if (BUILD_ONNX)
target_link_libraries(wav2wav onnxruntime SndFile::sndfile tokenizers_cpp tokenizers_c sentencepiece dl m pthread)
else()
if(CMAKE_CROSSCOMPILING)
target_link_libraries(wav2wav PUBLIC nncase.rt_modules.k230 Nncase.Runtime.Native functional_k230 mmz)
else()
target_link_libraries(wav2wav Nncase.Runtime.Native SndFile::sndfile tokenizers_cpp tokenizers_c sentencepiece dl m pthread)
endif()
endif()

add_executable(demo ${CMAKE_SOURCE_DIR}/demo/demo.cpp)
target_link_libraries(demo wav2wav)