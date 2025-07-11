SET(CMAKE_SYSTEM_NAME Windows)
SET(CMAKE_SYSTEM_PROCESSOR x86_64)

set(TARGET x86_64-pc-windows-msvc)

set(CMAKE_C_COMPILER_TARGET ${TARGET})
set(CMAKE_CXX_COMPILER_TARGET ${TARGET})

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} --target=${TARGET}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} --target=${TARGET}")

set(CMAKE_GENERATOR_TOOLSET "ClangCL")
