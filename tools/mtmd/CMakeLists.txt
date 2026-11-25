# mtmd
#
# llama.cpp-xdna includes only Qwen3-TTS audio generation and speaker encoding in libmtmd; vision, video and
# unrelated audio models remain disabled (see mtmd.cpp).

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd.h
            mtmd-helper.h
            models/models.h
            models/qwen3tts-spkenc.cpp
            models/qwen3tts-gen.cpp
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads vendor::miniaudio)
target_include_directories(mtmd PUBLIC  .)
target_compile_features   (mtmd PRIVATE cxx_std_17)
target_compile_definitions(mtmd PRIVATE _USE_MATH_DEFINES)

if (NOT MSVC)
    # for miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (TARGET BUILD_INFO)
    add_dependencies(mtmd BUILD_INFO)
endif()

# if mtmd is linked against llama-common, we throw an error
if (TARGET mtmd)
    get_target_property(libs mtmd LINK_LIBRARIES)
    if (libs AND "llama-common" IN_LIST libs)
        message(FATAL_ERROR "mtmd is designed to be a public library.\n"
                            "It must not link against llama-common")
    endif()
endif()
