# XDNA backend of llama.cpp-xdna, for the NPU of AMD Ryzen AI processors: its device layer (the direct access to
# the amdxdna driver of Linux, no XRT), what the host sends to the NPU with a command, and the backend of ggml that
# gives the multiplications of a model to the NPU. See doc/XDNA2.md.

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GGML_XDNA needs Linux: it uses the amdxdna driver")
endif()

# drm.h of the system, which the interface of the driver (uapi/amdxdna_accel.h) includes
find_path(XDNA_DRM_INCLUDE_DIR NAMES drm.h PATH_SUFFIXES drm libdrm REQUIRED)

add_library(xdna-device STATIC
            xdna-device.cpp
            xdna-device.h)

target_include_directories(xdna-device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# kernel headers, written in C: as system headers, so that the warnings of the project do not apply to them
target_include_directories(xdna-device SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/uapi ${XDNA_DRM_INCLUDE_DIR})
target_compile_features(xdna-device PRIVATE cxx_std_17)
set_target_properties(xdna-device PROPERTIES POSITION_INDEPENDENT_CODE ON)

# What the host sends to the NPU with a command: the instruction streams, the description of an overlay, the plan
# of a matrix multiplication and the conversion of its operands. It does not use the driver.
add_library(xdna-insts STATIC
            xdna-bfp16.cpp
            xdna-bfp16.h
            xdna-chain.cpp
            xdna-chain.h
            xdna-gemm.cpp
            xdna-gemm.h
            xdna-gdnp.cpp
            xdna-gdnp.h
            xdna-gemv.cpp
            xdna-gemv.h
            xdna-glu.cpp
            xdna-glu.h
            xdna-insts.cpp
            xdna-insts.h
            xdna-overlay.cpp
            xdna-overlay.h
            xdna-workers.cpp
            xdna-workers.h)

find_package(Threads REQUIRED)
target_link_libraries(xdna-insts PUBLIC Threads::Threads)
# With OpenMP, as the CPU backend of ggml has, the loops of the host side run on the threads of OpenMP: the ones the
# CPU backend uses, awake between its parallel regions (see xdna-workers.h)
if (GGML_OPENMP_ENABLED)
    target_compile_definitions(xdna-insts PUBLIC XDNA_OPENMP)
    target_link_libraries(xdna-insts PUBLIC ${GGML_OPENMP_TARGET_CXX})
endif()
target_include_directories(xdna-insts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(xdna-insts PRIVATE cxx_std_17)
set_target_properties(xdna-insts PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Matrix multiplications on the NPU, from floats to floats: the two libraries above together.
add_library(xdna-runner STATIC
            xdna-runner.cpp
            xdna-runner.h)

target_include_directories(xdna-runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xdna-runner PUBLIC xdna-device xdna-insts)
target_compile_features(xdna-runner PRIVATE cxx_std_17)
set_target_properties(xdna-runner PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The backend of ggml: it gives the multiplications of a model by its matrices of weights to the NPU.
ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         ../../include/ggml-xdna.h)

target_link_libraries(ggml-xdna PRIVATE xdna-runner)

# llama.cpp-xdna: the overlays the backend loads, which it looks for in the folder of the program (or in
# GGML_XDNA_OVERLAY_DIR): the one of the prompts and the chained one for generating tokens, each a PDI and its
# manifest. overlays/ keeps them built from overlays/designs/ and kernels/ (overlays/build-overlays.sh, byte for byte),
# so that nothing more is needed; with GGML_XDNA_BUILD_OVERLAYS, CMake builds them instead, with the mlir-aie and the
# Peano of the virtual environment GGML_XDNA_MLIR_AIE_VENV. They go next to the programs and are installed with them.
option(GGML_XDNA_BUILD_OVERLAYS "ggml: build the overlays of the XDNA backend with mlir-aie and Peano" OFF)
set(GGML_XDNA_MLIR_AIE_VENV "" CACHE PATH "ggml: virtual environment of Python with mlir_aie and llvm-aie")

set(XDNA_OVERLAY_FILES
    xdna2-gemm-bfp16.pdi
    xdna2-gemm-bfp16.manifest
    xdna2-gemv-chain.pdi
    xdna2-gemv-chain.manifest)

if (GGML_XDNA_BUILD_OVERLAYS)
    if (NOT GGML_XDNA_MLIR_AIE_VENV)
        message(FATAL_ERROR "GGML_XDNA_BUILD_OVERLAYS needs GGML_XDNA_MLIR_AIE_VENV: a virtual environment of Python with mlir_aie and llvm-aie")
    endif()
    set(XDNA_OVERLAY_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/overlays)
    set(XDNA_OVERLAY_BUILT)
    foreach (vFile ${XDNA_OVERLAY_FILES})
        list(APPEND XDNA_OVERLAY_BUILT ${XDNA_OVERLAY_SOURCE}/${vFile})
    endforeach()
    file(GLOB XDNA_OVERLAY_INPUTS
         ${CMAKE_CURRENT_SOURCE_DIR}/overlays/designs/*.py
         ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cc)
    add_custom_command(OUTPUT ${XDNA_OVERLAY_BUILT}
                       COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/overlays/build-overlays.sh ${GGML_XDNA_MLIR_AIE_VENV}
                               ${XDNA_OVERLAY_SOURCE}
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/overlays/build-overlays.sh ${XDNA_OVERLAY_INPUTS}
                       COMMENT "Building the overlays of the XDNA backend with mlir-aie"
                       VERBATIM)
else()
    set(XDNA_OVERLAY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/overlays)
endif()

# the folder of the programs
if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(XDNA_PROGRAMS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
    set(XDNA_PROGRAMS_DIR ${CMAKE_BINARY_DIR}/bin)
endif()

set(XDNA_OVERLAY_COPIES)
foreach (vFile ${XDNA_OVERLAY_FILES})
    add_custom_command(OUTPUT ${XDNA_PROGRAMS_DIR}/${vFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory ${XDNA_PROGRAMS_DIR}
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XDNA_OVERLAY_SOURCE}/${vFile}
                               ${XDNA_PROGRAMS_DIR}/${vFile}
                       DEPENDS ${XDNA_OVERLAY_SOURCE}/${vFile}
                       VERBATIM)
    list(APPEND XDNA_OVERLAY_COPIES ${XDNA_PROGRAMS_DIR}/${vFile})
    install(FILES ${XDNA_OVERLAY_SOURCE}/${vFile} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()
add_custom_target(xdna-overlays ALL DEPENDS ${XDNA_OVERLAY_COPIES})
add_dependencies(ggml-xdna xdna-overlays)
