# Splits the ggml-cuda kernel compile across machines.
#
# ggml-cuda is ~150 nvcc translation units, and nvcc compiles each of them once
# per GPU architecture in CMAKE_CUDA_ARCHITECTURES: roughly one core-hour per
# architecture, which is nearly all of a CUDA build. It only parallelizes across
# files, so CI fans the files out over several runners:
#
#   LLAMA_NODE_CUDA_SHARD=<i>/<n>
#     Compile slice i of n (1/4 ... 4/4) of the kernel sources and nothing
#     else: build the `cuda-objects` target and take the object files from
#     ${CMAKE_BINARY_DIR}/cuda-objects.
#
#   LLAMA_NODE_CUDA_OBJECTS_DIR=<dir>
#     Use the object files below <dir> (every slice, gathered in one place)
#     instead of compiling the kernel sources.
#
# Both sides must be configured with the same options otherwise; the objects are
# matched to the sources by name and a missing or unexpected one is an error.
#
# One kernel source, the "anchor", is left out of every slice and is always
# compiled by the consuming build. That keeps CUDA among the languages of the
# target, so whatever links it gets the same CUDA runtime libraries as in a
# build that compiles everything itself.

function(llama_node_cuda_shards target)
  if (LLAMA_NODE_CUDA_SHARD AND LLAMA_NODE_CUDA_OBJECTS_DIR)
    message(FATAL_ERROR "LLAMA_NODE_CUDA_SHARD and LLAMA_NODE_CUDA_OBJECTS_DIR are mutually exclusive")
  endif()
  if (NOT TARGET ${target})
    message(FATAL_ERROR "LLAMA_NODE_CUDA_SHARD / LLAMA_NODE_CUDA_OBJECTS_DIR need the ${target} target (GGML_CUDA=ON)")
  endif()

  get_target_property(sources ${target} SOURCES)
  set(kernels ${sources})
  list(FILTER kernels INCLUDE REGEX "\\.cu$")
  set(others ${sources})
  list(FILTER others EXCLUDE REGEX "\\.cu$")

  # Every machine has to agree on the order, whatever order the globs returned.
  list(SORT kernels)
  list(LENGTH kernels kernel_count)
  if (kernel_count LESS 2)
    message(FATAL_ERROR "${target} has ${kernel_count} .cu sources, nothing to split")
  endif()
  list(POP_FRONT kernels anchor)

  # Object files are matched to sources by name, so names have to be unique.
  set(kernel_names "")
  foreach (source IN LISTS kernels)
    get_filename_component(name "${source}" NAME_WLE)
    list(APPEND kernel_names "${name}")
  endforeach()
  set(unique_names ${kernel_names})
  list(REMOVE_DUPLICATES unique_names)
  list(LENGTH kernel_names name_count)
  list(LENGTH unique_names unique_count)
  if (NOT name_count EQUAL unique_count)
    message(FATAL_ERROR "${target} has .cu sources that share a file name; cuda-shards.cmake cannot tell their objects apart")
  endif()

  if (LLAMA_NODE_CUDA_SHARD)
    if (NOT LLAMA_NODE_CUDA_SHARD MATCHES "^([0-9]+)/([0-9]+)$")
      message(FATAL_ERROR "LLAMA_NODE_CUDA_SHARD must be <index>/<count>, got: ${LLAMA_NODE_CUDA_SHARD}")
    endif()
    set(shard_index ${CMAKE_MATCH_1})
    set(shard_count ${CMAKE_MATCH_2})
    if (shard_index LESS 1 OR shard_index GREATER shard_count)
      message(FATAL_ERROR "LLAMA_NODE_CUDA_SHARD index must be between 1 and the count, got: ${LLAMA_NODE_CUDA_SHARD}")
    endif()

    # Round-robin over the sorted names: the expensive template instances sit
    # next to each other, so every slice gets its share of each family.
    set(slice "")
    set(position 0)
    foreach (source IN LISTS kernels)
      math(EXPR position "${position} % ${shard_count} + 1")
      if (position EQUAL shard_index)
        list(APPEND slice "${source}")
      endif()
    endforeach()
    list(LENGTH slice slice_count)
    if (slice_count EQUAL 0)
      message(FATAL_ERROR "LLAMA_NODE_CUDA_SHARD=${LLAMA_NODE_CUDA_SHARD} is an empty slice of ${name_count} sources")
    endif()

    set_property(TARGET ${target} PROPERTY SOURCES ${others} ${slice})

    # Object file locations differ per generator, so gather them at build time
    # from the binary directory of the target, which holds nothing else.
    get_target_property(binary_dir ${target} BINARY_DIR)
    set(objects_dir "${CMAKE_BINARY_DIR}/cuda-objects")
    set(collect_script "${CMAKE_BINARY_DIR}/collect-cuda-objects.cmake")
    file(WRITE "${collect_script}" "\
file(GLOB_RECURSE objects \"${binary_dir}/*.o\" \"${binary_dir}/*.obj\")
list(LENGTH objects object_count)
if (NOT object_count EQUAL ${slice_count})
  message(FATAL_ERROR \"Expected ${slice_count} object files below ${binary_dir}, found \${object_count}\")
endif()
file(REMOVE_RECURSE \"${objects_dir}\")
file(MAKE_DIRECTORY \"${objects_dir}\")
file(COPY \${objects} DESTINATION \"${objects_dir}\")
message(STATUS \"Collected \${object_count} CUDA objects in ${objects_dir}\")
")
    add_custom_target(cuda-objects
      COMMAND ${CMAKE_COMMAND} -P "${collect_script}"
      COMMENT "Collecting CUDA objects of slice ${LLAMA_NODE_CUDA_SHARD}"
      VERBATIM
    )
    add_dependencies(cuda-objects ${target})

    message(STATUS "CUDA shard ${LLAMA_NODE_CUDA_SHARD}: ${slice_count} of ${name_count} kernel sources")
  else()
    get_filename_component(objects_dir "${LLAMA_NODE_CUDA_OBJECTS_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    file(GLOB_RECURSE objects "${objects_dir}/*.o" "${objects_dir}/*.obj")

    set(object_names "")
    foreach (object IN LISTS objects)
      get_filename_component(name "${object}" NAME_WLE)
      # Ninja and Makefiles keep the source extension (x.cu.o), Visual Studio does not (x.obj)
      string(REGEX REPLACE "\\.cu$" "" name "${name}")
      list(APPEND object_names "${name}")
    endforeach()

    set(missing ${kernel_names})
    set(unexpected ${object_names})
    if (object_names)
      list(REMOVE_ITEM missing ${object_names})
      list(REMOVE_ITEM unexpected ${kernel_names})
    endif()
    set(unique_names ${object_names})
    list(REMOVE_DUPLICATES unique_names)
    list(LENGTH object_names object_count)
    list(LENGTH unique_names unique_count)
    if (missing OR unexpected OR NOT object_count EQUAL unique_count)
      list(LENGTH missing missing_count)
      message(FATAL_ERROR
        "${objects_dir} does not hold exactly one object per ${target} kernel source "
        "(${object_count} objects for ${name_count} sources).\n"
        "  missing (${missing_count}): ${missing}\n"
        "  unexpected: ${unexpected}\n"
        "Were all slices built from this commit with the same options?")
    endif()

    set_property(TARGET ${target} PROPERTY SOURCES ${others} ${anchor} ${objects})
    set_source_files_properties(${objects} TARGET_DIRECTORY ${target} PROPERTIES EXTERNAL_OBJECT TRUE)

    message(STATUS "CUDA objects: using ${object_count} prebuilt kernel objects from ${objects_dir}")
  endif()
endfunction()
