cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project (llama-node)

if(NOT DEFINED napi_build_version)
  set(napi_build_version 6)
endif()
add_definitions(-DNAPI_VERSION=${napi_build_version})
message(STATUS "NAPI_VERSION: ${napi_build_version}")

# default build Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

include_directories(${CMAKE_JS_INC})

# flags: -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLAMA_STATIC ON CACHE BOOL "Build llama as static library")
add_subdirectory("src/llama.cpp")

file(GLOB SOURCE_FILES "src/addons.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

# static link llama and ggml
target_link_libraries(${PROJECT_NAME} PUBLIC llama ggml common)

string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
string(REPLACE "amd64" "x64" ARCH ${ARCH})
string(REPLACE "x86_64" "x64" ARCH ${ARCH})
string(REPLACE "i686" "x86" ARCH ${ARCH})
string(REPLACE "i386" "x86" ARCH ${ARCH})
string(REPLACE "arm" "arm" ARCH ${ARCH})
string(REPLACE "aarch64" "arm64" ARCH ${ARCH})

if(DEFINED VARIANT)
  string(APPEND PLATFORM "-${VARIANT}")
endif()

set(PLATFORM_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/${PLATFORM}/${ARCH})
message(STATUS "PLATFORM_BINARY_DIR: ${PLATFORM_BINARY_DIR}")

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  # copy target to bin folder
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_JS_NODELIB_TARGET} ${PLATFORM_BINARY_DIR}/${CMAKE_JS_NODELIB_TARGET}
    COMMENT "Copying to bin folder"
  )
endif()

# copy target to bin folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${PLATFORM_BINARY_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
    COMMENT "Copying to bin folder"
)
