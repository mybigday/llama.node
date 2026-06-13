cmake_minimum_required(VERSION 3.15)

project(llama_node_wasm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLAMA_INSTALL_VERSION "0.0.0")

option(LLAMA_NODE_WASM_WEBGPU "Build llama.node WASM with ggml WebGPU backend" ON)
option(LLAMA_NODE_WASM_THREADS "Build llama.node WASM with pthread support" OFF)
set(LLAMA_NODE_WASM_PTHREAD_POOL_SIZE "8" CACHE STRING "Emscripten pthread worker pool size")

set(LLAMA_CPP_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/llama.cpp.patch)

execute_process(
  COMMAND git apply --check ${LLAMA_CPP_PATCH_FILE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
  RESULT_VARIABLE LLAMA_CPP_PATCH_CHECK_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)

if (LLAMA_CPP_PATCH_CHECK_RESULT EQUAL 0)
  execute_process(
    COMMAND git apply ${LLAMA_CPP_PATCH_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
    RESULT_VARIABLE LLAMA_CPP_PATCH_APPLY_RESULT
  )

  if (NOT LLAMA_CPP_PATCH_APPLY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to apply scripts/llama.cpp.patch")
  endif()

  message(STATUS "Applied scripts/llama.cpp.patch")
else()
  execute_process(
    COMMAND git apply --reverse --check ${LLAMA_CPP_PATCH_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
    RESULT_VARIABLE LLAMA_CPP_PATCH_REVERSE_CHECK_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )

  if (LLAMA_CPP_PATCH_REVERSE_CHECK_RESULT EQUAL 0)
    message(STATUS "scripts/llama.cpp.patch already applied")
  else()
    message(FATAL_ERROR "scripts/llama.cpp.patch does not match the current src/llama.cpp checkout")
  endif()
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common")
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build tools")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
set(LLAMA_CURL OFF CACHE BOOL "Build curl")
set(LLAMA_OPENSSL OFF CACHE BOOL "Use OpenSSL" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for WASM" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "Disable mtmd video helpers for WASM" FORCE)

if (LLAMA_NODE_WASM_WEBGPU)
  set(GGML_WEBGPU ON CACHE BOOL "ggml: use WebGPU" FORCE)
  set(GGML_WEBGPU_JSPI ON CACHE BOOL "ggml: use JSPI for WebGPU" FORCE)
  add_link_options(-sJSPI_EXPORTS=llama_node_wasm_start,llama_node_wasm_action,llama_node_wasm_release_all)
else()
  set(GGML_WEBGPU OFF CACHE BOOL "ggml: use WebGPU" FORCE)
  set(GGML_WEBGPU_JSPI OFF CACHE BOOL "ggml: use JSPI for WebGPU" FORCE)
endif()

add_compile_options(
  -O3
  -msimd128
  -sMEMORY64=1
  -DNDEBUG
  -fwasm-exceptions
  -frtti
)

if (LLAMA_NODE_WASM_THREADS)
  add_compile_options(-pthread)
  add_link_options(-pthread -sUSE_PTHREADS=1 -sPTHREAD_POOL_SIZE=${LLAMA_NODE_WASM_PTHREAD_POOL_SIZE})
endif()

add_link_options(
  -O3
  --no-entry
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sENVIRONMENT=web,worker
  -sALLOW_MEMORY_GROWTH=1
  -sINITIAL_MEMORY=128MB
  -sMAXIMUM_MEMORY=4096MB
  -sMEMORY64=1
  -fwasm-exceptions
  -sFORCE_FILESYSTEM=1
  -sNO_EXIT_RUNTIME=1
  -sEXPORTED_FUNCTIONS=_llama_node_wasm_start,_llama_node_wasm_action,_llama_node_wasm_action_sync,_llama_node_wasm_release_all
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,HEAPU8
)

add_subdirectory("../llama.cpp" "${CMAKE_BINARY_DIR}/llama.cpp")
add_subdirectory("../llama.cpp/tools/mtmd" "${CMAKE_BINARY_DIR}/mtmd")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/tools/mtmd
  ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor
)

file(
  GLOB RN_LLAMA_SOURCE_FILES
    "../rn-llama/*.c"
    "../rn-llama/*.cpp"
)

add_executable(llama-node
  "llama-node-wasm.cpp"
  ${RN_LLAMA_SOURCE_FILES}
)

target_link_libraries(llama-node PRIVATE llama ggml llama-common mtmd)

set_target_properties(llama-node PROPERTIES
  OUTPUT_NAME "llama-node"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
