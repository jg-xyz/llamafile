cmake_minimum_required(VERSION 3.16)
project(Turbo1Bit C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ── Turbo1Bit core library ──────────────────────────────────────────

add_library(turbo1bit STATIC
    src/turbo1bit_codebook.c
    src/turbo1bit_rotation.c
    src/turbo1bit_quantizer.c
    src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit PUBLIC src)
target_link_libraries(turbo1bit PRIVATE m)

if(APPLE)
    target_link_libraries(turbo1bit PRIVATE "-framework Accelerate")
endif()

# ── Metal acceleration (macOS only) ─────────────────────────────────

if(APPLE)
    enable_language(OBJC)
    add_library(turbo1bit-metal STATIC src/turbo1bit_metal.m)
    target_include_directories(turbo1bit-metal PUBLIC src)
    target_link_libraries(turbo1bit-metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    set_source_files_properties(src/turbo1bit_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# ── Standalone benchmark executables ───────────────────────────────
# These do NOT link against llama.cpp — they test the TurboQuant
# algorithm in isolation.

add_executable(turbo1bit-bench tools/turbo1bit/turbo1bit_bench.c)
target_link_libraries(turbo1bit-bench PRIVATE turbo1bit m)
target_include_directories(turbo1bit-bench PRIVATE src)

add_executable(turbo1bit-stress tools/turbo1bit/turbo1bit_stress.c)
target_link_libraries(turbo1bit-stress PRIVATE turbo1bit m)
target_include_directories(turbo1bit-stress PRIVATE src)

if(APPLE)
    target_link_libraries(turbo1bit-bench PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-stress PRIVATE "-framework Accelerate")
endif()

# ── turbo1bit-infer (requires llama.cpp) ──────────────────────────
# Only built if llama.cpp is available as an external or parent project.

if(DEFINED LLAMA_CPP_DIR OR TARGET llama)
    if(NOT TARGET llama)
        add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)
    endif()

    add_executable(turbo1bit-infer tools/turbo1bit/turbo1bit_infer.cpp)
    target_include_directories(turbo1bit-infer PRIVATE
        src
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/src
        ${LLAMA_CPP_DIR}/common
    )
    target_link_libraries(turbo1bit-infer PRIVATE
        turbo1bit llama common m
    )
    if(APPLE)
        target_link_libraries(turbo1bit-infer PRIVATE
            turbo1bit-metal
            "-framework Metal"
            "-framework Foundation"
            "-framework Accelerate"
        )
    endif()
else()
    message(STATUS "Turbo1Bit: llama.cpp not found — skipping turbo1bit-infer.")
    message(STATUS "  Pass -DLLAMA_CPP_DIR=/path/to/llama.cpp to enable.")
endif()
