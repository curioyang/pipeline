cmake_minimum_required(VERSION 3.5)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(wav2wav)

#option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)
#option(BUILD_JNI "Build JNI for android app." OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sndfile/lib/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})
find_package(SndFile REQUIRED)

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


# onnxruntime
set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)
link_directories(${ONNXRUNTIME_PATH}/lib)

# librosa
set(ROSA_PATH ${CMAKE_SOURCE_DIR}/3rd_party/librosa)

# cnpy
set(CNPY_PATH ${CMAKE_SOURCE_DIR}/3rd_party/libcnpy)
link_directories(${CNPY_PATH}/lib)
include_directories(${CNPY_PATH}/include)

include_directories(${ROSA_PATH}
                    ${ONNXRUNTIME_PATH}/include
                    ${CMAKE_SOURCE_DIR}/include)

FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(wav2wav STATIC ${SRCS}
        src/audio.cpp
        src/utils.cpp)
target_link_libraries(wav2wav onnxruntime SndFile::sndfile fftw3 cnpy)

add_executable(demo ${CMAKE_SOURCE_DIR}/demo/demo.cpp)
target_link_libraries(demo wav2wav)