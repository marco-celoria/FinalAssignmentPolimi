cmake_minimum_required(VERSION 3.18...3.30)

project(CoolingSimulation LANGUAGES CXX)

# ============================================================
# Build Options
# ============================================================

set(COOLING_BUILD_OPENMP "ON" CACHE STRING
    "Build the OpenMP C++ solver: ON or OFF"
)

set(COOLING_BUILD_CUDA "AUTO" CACHE STRING
    "Build the CUDA C++ solver: ON, OFF, or AUTO"
)

set(COOLING_CUDA_ARCHITECTURES "AUTO" CACHE STRING
    "CUDA architectures: AUTO, native, 70, 75, 80, 86, 89, 90, or a list like 70;80;90"
)

set_property(CACHE COOLING_BUILD_OPENMP PROPERTY STRINGS ON OFF)
set_property(CACHE COOLING_BUILD_CUDA PROPERTY STRINGS ON OFF AUTO)

option(COOLING_NATIVE_ARCH
    "Enable native host architecture flags"
    OFF
)

option(COOLING_FAST_MATH_CUDA
    "Enable --use_fast_math for CUDA code"
    OFF
)

option(COOLING_STRICT_CUDA
    "Fail if CUDA was requested but unavailable"
    OFF
)

# ------------------------------------------------------------
# Backward Compatibility Configuration
# ------------------------------------------------------------

if(DEFINED PARTICLES_BUILD_OPENMP)
    set(COOLING_BUILD_OPENMP "${PARTICLES_BUILD_OPENMP}" CACHE STRING "" FORCE)
endif()

if(DEFINED PARTICLES_BUILD_CUDA)
    set(COOLING_BUILD_CUDA "${PARTICLES_BUILD_CUDA}" CACHE STRING "" FORCE)
endif()

if(DEFINED PARTICLES_NATIVE_ARCH)
    set(COOLING_NATIVE_ARCH "${PARTICLES_NATIVE_ARCH}" CACHE BOOL "" FORCE)
endif()

if(DEFINED PARTICLES_FAST_MATH_CUDA)
    set(COOLING_FAST_MATH_CUDA "${PARTICLES_FAST_MATH_CUDA}" CACHE BOOL "" FORCE)
endif()

if(DEFINED PARTICLES_CUDA_ARCHITECTURES)
    set(COOLING_CUDA_ARCHITECTURES "${PARTICLES_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
endif()

# ============================================================
# Validate Options
# ============================================================

if(NOT COOLING_BUILD_OPENMP MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "COOLING_BUILD_OPENMP must be ON or OFF")
endif()

if(NOT COOLING_BUILD_CUDA MATCHES "^(ON|OFF|AUTO)$")
    message(FATAL_ERROR "COOLING_BUILD_CUDA must be ON, OFF, or AUTO")
endif()

# Native macOS cannot build the CUDA C++ target.
if(APPLE AND NOT COOLING_BUILD_CUDA STREQUAL "OFF")
    message(WARNING
        "macOS detected. Native CUDA target is not supported here. "
        "Forcing COOLING_BUILD_CUDA=OFF."
    )
    set(COOLING_BUILD_CUDA "OFF" CACHE STRING "" FORCE)
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

if(COOLING_BUILD_OPENMP STREQUAL "ON")
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ============================================================
# CUDA Toolchain and Initialization
# ============================================================

set(COOLING_CUDA_AVAILABLE OFF)

if(NOT COOLING_BUILD_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT COOLING_CUDA_ARCHITECTURES STREQUAL "AUTO"
           AND NOT COOLING_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "${COOLING_CUDA_ARCHITECTURES}")
            message(STATUS "CUDA architectures explicitly set to: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            message(STATUS "CUDA architectures set to: AUTO / CMake default")
        endif()

        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        set(COOLING_CUDA_AVAILABLE ON)
    else()
        if(COOLING_BUILD_CUDA STREQUAL "ON" OR COOLING_STRICT_CUDA)
            message(FATAL_ERROR
                "CUDA was requested, but CMake could not find a CUDA compiler. "
                "Set CUDACXX, pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc, "
                "or configure with -DCOOLING_BUILD_CUDA=OFF."
            )
        else()
            message(WARNING
                "CUDA compiler not found. CUDA target will not be built. "
                "Use -DCOOLING_BUILD_CUDA=ON to make this a hard error."
            )
        endif()
    endif()
endif()

# ============================================================
# Helper Functions
# ============================================================

function(cooling_apply_hdf5 target_name)
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

function(cooling_apply_warnings target_name)
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

function(cooling_apply_optimization target_name)
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

        if(COOLING_NATIVE_ARCH)
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
            )
        endif()
    endif()
endfunction()

function(cooling_apply_fp_policy target_name)
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

    # CUDA fast math remains controlled separately by COOLING_FAST_MATH_CUDA.
    # Default is OFF.
endfunction()

function(cooling_apply_common_settings target_name)
    cooling_apply_hdf5(${target_name})
    cooling_apply_warnings(${target_name})
    cooling_apply_optimization(${target_name})
    cooling_apply_fp_policy(${target_name})
endfunction()

# ============================================================
# Targets
# ============================================================

# ------------------------------------------------------------
# OpenMP C++ solver
# ------------------------------------------------------------

if(COOLING_BUILD_OPENMP STREQUAL "ON")
    add_executable(cooling_omp
        src/cpp/cooling.cpp
    )

    target_compile_features(cooling_omp PRIVATE cxx_std_17)
    cooling_apply_common_settings(cooling_omp)

    target_link_libraries(cooling_omp
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()

# ------------------------------------------------------------
# CUDA C++ solver
# ------------------------------------------------------------

if(COOLING_CUDA_AVAILABLE)
    add_executable(cooling_cuda
        src/cpp/cooling.cu
    )

    target_compile_features(cooling_cuda PRIVATE cxx_std_17)
    cooling_apply_common_settings(cooling_cuda)

    set_target_properties(cooling_cuda
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    if(COOLING_FAST_MATH_CUDA)
        target_compile_options(cooling_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

# ============================================================
# Install
# ============================================================

if(TARGET cooling_omp)
    install(TARGETS cooling_omp
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET cooling_cuda)
    install(TARGETS cooling_cuda
        RUNTIME DESTINATION bin
    )
endif()

# ============================================================
# Configuration Summary
# ============================================================

message(STATUS "")
message(STATUS "CoolingSimulation configuration")
message(STATUS "  System:                    ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Processor:                 ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C++ compiler:              ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ compiler ID:           ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type:                ${CMAKE_BUILD_TYPE}")
message(STATUS "  HDF5 include dirs:         ${HDF5_INCLUDE_DIRS}")
message(STATUS "  HDF5 libraries:            ${HDF5_LIBRARIES}")
message(STATUS "  Build OpenMP solver:       ${COOLING_BUILD_OPENMP}")
message(STATUS "  Build CUDA solver:         ${COOLING_BUILD_CUDA}")
message(STATUS "  CUDA available:            ${COOLING_CUDA_AVAILABLE}")
message(STATUS "  Native host architecture:  ${COOLING_NATIVE_ARCH}")
message(STATUS "  FP policy:                 moderate (built-in)")

if(COOLING_BUILD_OPENMP STREQUAL "ON")
    message(STATUS "  OpenMP CXX flags:          ${OpenMP_CXX_FLAGS}")
    message(STATUS "  OpenMP CXX libraries:      ${OpenMP_CXX_LIBRARIES}")
endif()

if(COOLING_CUDA_AVAILABLE)
    message(STATUS "  CUDA compiler:             ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA architectures:        ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA fast math:            ${COOLING_FAST_MATH_CUDA}")
endif()

message(STATUS "")
