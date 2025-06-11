cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project (index)

set(CMAKE_CXX_STANDARD 17)

execute_process(COMMAND
  git apply ${CMAKE_CURRENT_SOURCE_DIR}/scripts/llama.cpp.patch
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT DEFINED napi_build_version)
  set(napi_build_version 6)
endif()
add_definitions(-DNAPI_VERSION=${napi_build_version})
message(STATUS "NAPI_VERSION: ${napi_build_version}")

set(CMAKE_SYSTEM_PROCESSOR ${NODE_ARCH})
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

# normalize platform to nodejs
string(REPLACE "windows" "win32" PLATFORM ${PLATFORM})

# normalize arch to nodejs: 'arm', 'arm64', 'ia32', 'loong64', 'mips', 'mipsel', 'ppc', 'ppc64', 'riscv64', 's390', 's390x', and 'x64'.
string(REPLACE "amd64" "x64" ARCH ${ARCH})
string(REPLACE "x86_64" "x64" ARCH ${ARCH})
string(REPLACE "i686" "ia32" ARCH ${ARCH})
string(REPLACE "i386" "ia32" ARCH ${ARCH})
string(REPLACE "armv7l" "arm" ARCH ${ARCH})
string(REPLACE "arm" "arm" ARCH ${ARCH})
string(REPLACE "arm64x" "arm64" ARCH ${ARCH})
string(REPLACE "aarch64" "arm64" ARCH ${ARCH})

option(TO_PACKAGE "Build as package" OFF)

if(DEFINED VARIANT)
  set(VARIANT -${VARIANT})
else()
  set(VARIANT "")
endif()

if (TO_PACKAGE)
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/packages/node-llama-${PLATFORM}-${ARCH}${VARIANT})
else()
  set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "PLATFORM_BINARY_DIR: ${PLATFORM_BINARY_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if((UNIX OR MINGW) AND NOT CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
  endif()
endif()

# Improve speed
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if (MSVC)
    if (NOT GGML_VULKAN)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Ob2 /Oi /Ot /Oy /GL")
      set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /LTCG")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O1 /Ob1 /bigobj")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O1 /Ob1 /bigobj")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -funroll-loops -flto")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -flto")
  endif()
endif()

if (CLANG)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# flags: -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# VULKAN_SDK
if (VULKAN_SDK)
  set(ENV{VULKAN_SDK} ${VULKAN_SDK})
  find_package(Vulkan REQUIRED)
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common")

set(LLAMA_CURL OFF CACHE BOOL "Build curl")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
add_subdirectory("src/llama.cpp")
add_subdirectory("src/llama.cpp/tools/mtmd")

include_directories(
  ${CMAKE_JS_INC}
  "src/llama.cpp"
  "src/llama.cpp/src"
  "src/tools/mtmd"
)

file(
  GLOB SOURCE_FILES
    "src/addons.cc"
    "src/common.hpp"
    "src/DisposeWorker.cpp"
    "src/DisposeWorker.h"
    "src/LlamaCompletionWorker.cpp"
    "src/LlamaCompletionWorker.h"
    "src/LlamaContext.cpp"
    "src/LlamaContext.h"
    "src/TokenizeWorker.cpp"
    "src/TokenizeWorker.h"
    "src/DetokenizeWorker.cpp"
    "src/DetokenizeWorker.h"
    "src/EmbeddingWorker.cpp"
    "src/EmbeddingWorker.h"
    "src/LoadSessionWorker.cpp"
    "src/LoadSessionWorker.h"
    "src/SaveSessionWorker.cpp"
    "src/SaveSessionWorker.h"
    "src/DecodeAudioTokenWorker.cpp"
    "src/DecodeAudioTokenWorker.h"
    "src/tts_utils.cpp"
    "src/tts_utils.h"
    "src/anyascii.h"
    "src/anyascii.c"
)

if (NOT MSVC AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(GLOB WIN_DYNAMIC_LOAD_SRC "src/win_dynamic_load.c")

  add_library(win_dynamic_load ${WIN_DYNAMIC_LOAD_SRC})
  set_target_properties(win_dynamic_load PROPERTIES COMPILE_FLAGS "-Wno-implicit-function-declaration")
  
  unset(CMAKE_JS_SRC)
  unset(CMAKE_JS_LIB)
  unset(CMAKE_JS_NODELIB_DEF)
  unset(CMAKE_JS_NODELIB_TARGET)
  unset(CMAKE_JS_NODELIB_TARGET_NAME)
  string(REGEX REPLACE "/DELAYLOAD:NODE.EXE" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

  set(CMAKE_JS_LIB win_dynamic_load)
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} llama ggml common mtmd ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(copy_assets ALL DEPENDS ${PROJECT_NAME})

if (TO_PACKAGE)
  add_custom_command(TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${PLATFORM_BINARY_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
    COMMENT "Deploy as package"
  )
endif()

if (TO_PACKAGE)
  set(METAL_LIB_TARGET_PATH ${PLATFORM_BINARY_DIR})
else()
  set(METAL_LIB_TARGET_PATH ${CMAKE_BINARY_DIR}/bin/default.metallib)
endif()

if(CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if (GGML_METAL AND NOT GGML_METAL_EMBED_LIBRARY)
  # copy ${CMAKE_BINARY_DIR}/bin/default.metallib  
  add_custom_command(
    TARGET copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/default.metallib ${METAL_LIB_TARGET_PATH}
    COMMENT "Copying default.metallib to bin folder"
  )
  add_dependencies(copy_assets ggml-metal)
endif()

if (GGML_CLBLAST AND TO_PACKAGE)
  find_package(CLBlast)
  if (CLBlast_FOUND)
    message(STATUS "CLBlast found: ${CLBlast_DIR}")
    file(
      GLOB CLBlast_SO_FILES
      ${CLBlast_DIR}/../../../bin/clblast.dll
      ${CLBlast_DIR}/../../../lib/libclblast.so
    )
    add_custom_command(
      TARGET copy_assets
      COMMAND ${CMAKE_COMMAND} -E copy ${CLBlast_SO_FILES} ${PLATFORM_BINARY_DIR}
      COMMENT "Copying CLBlast SO files to bin folder"
    )
  endif()
endif()
