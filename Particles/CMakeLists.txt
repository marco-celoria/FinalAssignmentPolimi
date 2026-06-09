cmake_minimum_required(VERSION 3.18...3.30)

project(ParticlesSimulation LANGUAGES CXX)

# ============================================================
# Build Options
# ============================================================

set(PARTICLES_BUILD_OPENMP "ON" CACHE STRING
    "Build the OpenMP C++ solver: ON or OFF"
)

set(PARTICLES_BUILD_CUDA "AUTO" CACHE STRING
    "Build the CUDA C++ solver: ON, OFF, or AUTO"
)

set(PARTICLES_CUDA_ARCHITECTURES "AUTO" CACHE STRING
    "CUDA architectures: AUTO, native, 70, 75, 80, 86, 89, 90, or a list like 70;80;90"
)

set_property(CACHE PARTICLES_BUILD_OPENMP PROPERTY STRINGS ON OFF)
set_property(CACHE PARTICLES_BUILD_CUDA PROPERTY STRINGS ON OFF AUTO)

option(PARTICLES_NATIVE_ARCH
    "Enable native host architecture flags"
    OFF
)

option(PARTICLES_FAST_MATH_CUDA
    "Enable --use_fast_math for CUDA code"
    OFF
)

option(PARTICLES_STRICT_CUDA
    "Fail if CUDA was requested but unavailable"
    OFF
)

# ------------------------------------------------------------
# Backward Compatibility Configuration
# ------------------------------------------------------------

if(DEFINED PARTICLES_BUILD_OPENMP)
    set(PARTICLES_BUILD_OPENMP "${PARTICLES_BUILD_OPENMP}" CACHE STRING "" FORCE)
endif()

if(DEFINED PARTICLES_BUILD_CUDA)
    set(PARTICLES_BUILD_CUDA "${PARTICLES_BUILD_CUDA}" CACHE STRING "" FORCE)
endif()

if(DEFINED PARTICLES_NATIVE_ARCH)
    set(PARTICLES_NATIVE_ARCH "${PARTICLES_NATIVE_ARCH}" CACHE BOOL "" FORCE)
endif()

if(DEFINED PARTICLES_FAST_MATH_CUDA)
    set(PARTICLES_FAST_MATH_CUDA "${PARTICLES_FAST_MATH_CUDA}" CACHE BOOL "" FORCE)
endif()

if(DEFINED PARTICLES_CUDA_ARCHITECTURES)
    set(PARTICLES_CUDA_ARCHITECTURES "${PARTICLES_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
endif()

# ============================================================
# Validate Options
# ============================================================

if(NOT PARTICLES_BUILD_OPENMP MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "PARTICLES_BUILD_OPENMP must be ON or OFF")
endif()

if(NOT PARTICLES_BUILD_CUDA MATCHES "^(ON|OFF|AUTO)$")
    message(FATAL_ERROR "PARTICLES_BUILD_CUDA must be ON, OFF, or AUTO")
endif()

# Native macOS cannot build the CUDA C++ target.
if(APPLE AND NOT PARTICLES_BUILD_CUDA STREQUAL "OFF")
    message(WARNING
        "macOS detected. Native CUDA target is not supported here. "
        "Forcing PARTICLES_BUILD_CUDA=OFF."
    )
    set(PARTICLES_BUILD_CUDA "OFF" CACHE STRING "" FORCE)
endif()

# ============================================================
# C++ Standard and Default Build Type
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# Dependencies
# ============================================================

find_package(HDF5 REQUIRED COMPONENTS C CXX)

if(PARTICLES_BUILD_OPENMP STREQUAL "ON")
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ============================================================
# CUDA Toolchain and Initialization
# ============================================================

set(PARTICLES_CUDA_AVAILABLE OFF)

if(NOT PARTICLES_BUILD_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT PARTICLES_CUDA_ARCHITECTURES STREQUAL "AUTO"
           AND NOT PARTICLES_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "${PARTICLES_CUDA_ARCHITECTURES}")
            message(STATUS "CUDA architectures explicitly set to: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            message(STATUS "CUDA architectures set to: AUTO / CMake default")
        endif()

        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        set(PARTICLES_CUDA_AVAILABLE ON)
    else()
        if(PARTICLES_BUILD_CUDA STREQUAL "ON" OR PARTICLES_STRICT_CUDA)
            message(FATAL_ERROR
                "CUDA was requested, but CMake could not find a CUDA compiler. "
                "Set CUDACXX, pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc, "
                "or configure with -DPARTICLES_BUILD_CUDA=OFF."
            )
        else()
            message(WARNING
                "CUDA compiler not found. CUDA target will not be built. "
                "Use -DPARTICLES_BUILD_CUDA=ON to make this a hard error."
            )
        endif()
    endif()
endif()

# ============================================================
# Helper Functions
# ============================================================

function(particles_apply_hdf5 target_name)
    if(TARGET hdf5::hdf5_cpp)
        if(TARGET hdf5::hdf5)
            target_link_libraries(${target_name}
                PRIVATE
                    hdf5::hdf5_cpp
                    hdf5::hdf5
            )
        else()
            target_link_libraries(${target_name}
                PRIVATE
                    hdf5::hdf5_cpp
            )
        endif()
    elseif(TARGET hdf5::hdf5)
        target_link_libraries(${target_name}
            PRIVATE
                hdf5::hdf5
        )
    else()
        target_include_directories(${target_name}
            PRIVATE
                ${HDF5_INCLUDE_DIRS}
        )

        target_compile_definitions(${target_name}
            PRIVATE
                ${HDF5_DEFINITIONS}
        )

        if(HDF5_CXX_LIBRARIES)
            target_link_libraries(${target_name}
                PRIVATE
                    ${HDF5_CXX_LIBRARIES}
                    ${HDF5_LIBRARIES}
            )
        else()
            target_link_libraries(${target_name}
                PRIVATE
                    ${HDF5_LIBRARIES}
            )
        endif()
    endif()
endfunction()

function(particles_apply_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-pedantic>
        )
    endif()
endfunction()

function(particles_apply_optimization target_name)
    if(MSVC)
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/O2>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
        )
    else()
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-O3>
                $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )

        if(PARTICLES_NATIVE_ARCH)
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
            )
        endif()
    endif()
endfunction()

function(particles_apply_fp_policy target_name)
    # Fixed "moderate" scientific FP policy:
    # - no fast math
    # - no unsafe math optimizations
    # - keep FMA contraction enabled on CPU for performance
    # - do not force strict CUDA math
    if(MSVC)
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:precise>
        )
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
                $<$<COMPILE_LANGUAGE:CXX>:-fno-unsafe-math-optimizations>
                $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>
        )
    endif()

    # CUDA fast math remains controlled separately by PARTICLES_FAST_MATH_CUDA.
    # Default is OFF.
endfunction()

function(particles_apply_common_settings target_name)
    particles_apply_hdf5(${target_name})
    particles_apply_warnings(${target_name})
    particles_apply_optimization(${target_name})
    particles_apply_fp_policy(${target_name})
endfunction()

# ============================================================
# Targets
# ============================================================

# ------------------------------------------------------------
# OpenMP C++ solver
# ------------------------------------------------------------

if(PARTICLES_BUILD_OPENMP STREQUAL "ON")
    add_executable(particles_omp
        src/cpp/particles.cpp
    )

    target_compile_features(particles_omp PRIVATE cxx_std_17)
    particles_apply_common_settings(particles_omp)

    target_link_libraries(particles_omp
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()

# ------------------------------------------------------------
# CUDA C++ solver
# ------------------------------------------------------------

if(PARTICLES_CUDA_AVAILABLE)
    add_executable(particles_cuda
        src/cpp/particles.cu
    )

    target_compile_features(particles_cuda PRIVATE cxx_std_17)
    particles_apply_common_settings(particles_cuda)

    set_target_properties(particles_cuda
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    if(PARTICLES_FAST_MATH_CUDA)
        target_compile_options(particles_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

# ============================================================
# Install
# ============================================================

if(TARGET particles_omp)
    install(TARGETS particles_omp
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET particles_cuda)
    install(TARGETS particles_cuda
        RUNTIME DESTINATION bin
    )
endif()

# ============================================================
# Configuration Summary
# ============================================================

message(STATUS "")
message(STATUS "ParticlesSimulation configuration")
message(STATUS "  System:                    ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Processor:                 ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C++ compiler:              ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ compiler ID:           ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type:                ${CMAKE_BUILD_TYPE}")
message(STATUS "  HDF5 include dirs:         ${HDF5_INCLUDE_DIRS}")
message(STATUS "  HDF5 libraries:            ${HDF5_LIBRARIES}")
message(STATUS "  Build OpenMP solver:       ${PARTICLES_BUILD_OPENMP}")
message(STATUS "  Build CUDA solver:         ${PARTICLES_BUILD_CUDA}")
message(STATUS "  CUDA available:            ${PARTICLES_CUDA_AVAILABLE}")
message(STATUS "  Native host architecture:  ${PARTICLES_NATIVE_ARCH}")
message(STATUS "  FP policy:                 moderate (built-in)")

if(PARTICLES_BUILD_OPENMP STREQUAL "ON")
    message(STATUS "  OpenMP CXX flags:          ${OpenMP_CXX_FLAGS}")
    message(STATUS "  OpenMP CXX libraries:      ${OpenMP_CXX_LIBRARIES}")
endif()

if(PARTICLES_CUDA_AVAILABLE)
    message(STATUS "  CUDA compiler:             ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA architectures:        ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA fast math:            ${PARTICLES_FAST_MATH_CUDA}")
endif()

message(STATUS "")
