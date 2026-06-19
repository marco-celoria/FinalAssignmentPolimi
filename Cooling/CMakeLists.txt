cmake_minimum_required(VERSION 3.25...3.30)
project(CoolingSimulation LANGUAGES CXX)

# ============================================================
# Options
# ============================================================

set(COOLING_BUILD_HDF5 "AUTO" CACHE STRING
    "Build HDF5 support: OFF, ON, or AUTO"
)

set(COOLING_BUILD_OPENMP "OFF" CACHE STRING
    "Build OpenMP executable: OFF, ON, or AUTO"
)

set(COOLING_BUILD_CUDA "OFF" CACHE STRING
    "Build CUDA executable: OFF, ON, or AUTO"
)

set(COOLING_CUDA_ARCHITECTURES "AUTO" CACHE STRING
    "CUDA architectures: AUTO, native, 80, 90, or list like 70;80;90"
)

option(COOLING_NATIVE_ARCH
    "Enable native host architecture flags"
    OFF
)

option(COOLING_FAST_MATH_CUDA
    "Enable --use_fast_math for CUDA code"
    OFF
)

option(COOLING_STRICT_CUDA
    "Fail if CUDA is requested but unavailable"
    OFF
)

set_property(CACHE COOLING_BUILD_HDF5 PROPERTY STRINGS OFF ON AUTO)
set_property(CACHE COOLING_BUILD_OPENMP PROPERTY STRINGS OFF ON AUTO)
set_property(CACHE COOLING_BUILD_CUDA PROPERTY STRINGS OFF ON AUTO)

# ============================================================
# Validate options
# ============================================================

if(NOT COOLING_BUILD_HDF5 MATCHES "^(OFF|ON|AUTO)$")
    message(FATAL_ERROR "COOLING_BUILD_HDF5 must be OFF, ON, or AUTO")
endif()

if(NOT COOLING_BUILD_OPENMP MATCHES "^(OFF|ON|AUTO)$")
    message(FATAL_ERROR "COOLING_BUILD_OPENMP must be OFF, ON, or AUTO")
endif()

if(NOT COOLING_BUILD_CUDA MATCHES "^(OFF|ON|AUTO)$")
    message(FATAL_ERROR "COOLING_BUILD_CUDA must be OFF, ON, or AUTO")
endif()

if(APPLE AND NOT COOLING_BUILD_CUDA STREQUAL "OFF")
    message(WARNING
        "macOS detected. CUDA is not supported on native Apple Silicon/macOS. "
        "Forcing COOLING_BUILD_CUDA=OFF."
    )
    set(COOLING_BUILD_CUDA "OFF" CACHE STRING "" FORCE)
endif()

# ============================================================
# Language standard and default build type
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# Optional HDF5
# ============================================================

set(COOLING_HDF5_AVAILABLE OFF)

if(NOT COOLING_BUILD_HDF5 STREQUAL "OFF")
    find_package(HDF5 QUIET COMPONENTS C CXX)

    if(HDF5_FOUND)
        set(COOLING_HDF5_AVAILABLE ON)
    elseif(COOLING_BUILD_HDF5 STREQUAL "ON")
        message(FATAL_ERROR
            "HDF5 support was requested with COOLING_BUILD_HDF5=ON, "
            "but HDF5 was not found."
        )
    else()
        message(WARNING
            "HDF5 not found. Building without HDF5 support."
        )
    endif()
endif()

# ============================================================
# Optional OpenMP
# ============================================================

set(COOLING_OPENMP_AVAILABLE OFF)

if(NOT COOLING_BUILD_OPENMP STREQUAL "OFF")
    find_package(OpenMP QUIET COMPONENTS CXX)

    if(OpenMP_CXX_FOUND)
        set(COOLING_OPENMP_AVAILABLE ON)
    elseif(COOLING_BUILD_OPENMP STREQUAL "ON")
        message(FATAL_ERROR
            "OpenMP was requested, but OpenMP was not found."
        )
    else()
        message(WARNING
            "OpenMP not found. OpenMP target will not be built."
        )
    endif()
endif()

# ============================================================
# Optional CUDA
# ============================================================

set(COOLING_CUDA_AVAILABLE OFF)

if(NOT COOLING_BUILD_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        if(NOT COOLING_CUDA_ARCHITECTURES STREQUAL "AUTO"
           AND NOT COOLING_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "${COOLING_CUDA_ARCHITECTURES}")
        endif()

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        set(COOLING_CUDA_AVAILABLE ON)
    elseif(COOLING_BUILD_CUDA STREQUAL "ON" OR COOLING_STRICT_CUDA)
        message(FATAL_ERROR
            "CUDA was requested, but no CUDA compiler was found."
        )
    else()
        message(WARNING
            "CUDA compiler not found. CUDA target will not be built."
        )
    endif()
endif()

# ============================================================
# Source files
# ============================================================

set(COOLING_SERIAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/cooling.cpp")
set(COOLING_OMP_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/cooling_omp.cpp")
set(COOLING_CUDA_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/cooling.cu")

if(NOT EXISTS "${COOLING_SERIAL_SRC}")
    message(FATAL_ERROR "Serial source file not found: ${COOLING_SERIAL_SRC}")
endif()

# ============================================================
# Helper functions
# ============================================================


function(cooling_apply_hdf5 target_name)
	if(NOT COOLING_HDF5_AVAILABLE)
        return()
    endif()

    target_compile_definitions(${target_name} PRIVATE USE_HDF5=1)

    if(TARGET hdf5::hdf5_cpp)
        target_link_libraries(${target_name} PRIVATE hdf5::hdf5_cpp)

        if(TARGET hdf5::hdf5)
            target_link_libraries(${target_name} PRIVATE hdf5::hdf5)
        endif()
    elseif(TARGET HDF5::HDF5)
        target_link_libraries(${target_name} PRIVATE HDF5::HDF5)
    elseif(TARGET hdf5::hdf5)
        target_link_libraries(${target_name} PRIVATE hdf5::hdf5)
    else()
        target_include_directories(${target_name} PRIVATE ${HDF5_INCLUDE_DIRS})
        target_compile_definitions(${target_name} PRIVATE ${HDF5_DEFINITIONS})

        if(HDF5_CXX_LIBRARIES)
            target_link_libraries(${target_name} PRIVATE ${HDF5_CXX_LIBRARIES})
        endif()

        if(HDF5_LIBRARIES)
            target_link_libraries(${target_name} PRIVATE ${HDF5_LIBRARIES})
        endif()
    endif()
endfunction()

function(cooling_apply_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-pedantic>
        )
    endif()
endfunction()

function(cooling_apply_native_arch target_name)
    if(COOLING_NATIVE_ARCH AND NOT MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
        )
    endif()
endfunction()

function(cooling_apply_common_settings target_name)
    cooling_apply_hdf5(${target_name})
    cooling_apply_warnings(${target_name})
    cooling_apply_native_arch(${target_name})
endfunction()

function(cooling_optional_source target_name source_path mode label)
    if(NOT EXISTS "${source_path}")
        if(mode STREQUAL "ON")
            message(FATAL_ERROR
                "${label} was requested, but source file is missing: ${source_path}"
            )
        else()
            message(WARNING
                "${label} source file not found. Target will not be built: ${source_path}"
            )
            return()
        endif()
    endif()

    add_executable(${target_name} "${source_path}")
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    cooling_apply_common_settings(${target_name})
endfunction()

# ============================================================
# Targets
# ============================================================

# Serial target: always built
add_executable(cooling_serial "${COOLING_SERIAL_SRC}")
target_compile_features(cooling_serial PRIVATE cxx_std_17)
cooling_apply_common_settings(cooling_serial)
target_compile_definitions(cooling_serial PRIVATE COOLING_SERIAL=1)

# OpenMP target: optional
if(COOLING_OPENMP_AVAILABLE)
    cooling_optional_source(
        cooling_omp
        "${COOLING_OMP_SRC}"
        "${COOLING_BUILD_OPENMP}"
        "OpenMP"
    )

    if(TARGET cooling_omp)
        target_link_libraries(cooling_omp PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(cooling_omp PRIVATE COOLING_OPENMP=1)
    endif()
endif()

# CUDA target: optional
if(COOLING_CUDA_AVAILABLE)
    cooling_optional_source(
        cooling_cuda
        "${COOLING_CUDA_SRC}"
        "${COOLING_BUILD_CUDA}"
        "CUDA"
    )

    if(TARGET cooling_cuda)
        set_target_properties(cooling_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        if(COOLING_FAST_MATH_CUDA)
            target_compile_options(cooling_cuda PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            )
        endif()

        target_compile_definitions(cooling_cuda PRIVATE COOLING_CUDA=1)
    endif()
endif()

# ============================================================
# Install
# ============================================================

install(TARGETS cooling_serial RUNTIME DESTINATION bin)

if(TARGET cooling_omp)
    install(TARGETS cooling_omp RUNTIME DESTINATION bin)
endif()

if(TARGET cooling_cuda)
    install(TARGETS cooling_cuda RUNTIME DESTINATION bin)
endif()

# ============================================================
# Summary
# ============================================================

if(TARGET cooling_omp)
    set(COOLING_OMP_TARGET_BUILT ON)
else()
    set(COOLING_OMP_TARGET_BUILT OFF)
endif()

if(TARGET cooling_cuda)
    set(COOLING_CUDA_TARGET_BUILT ON)
else()
    set(COOLING_CUDA_TARGET_BUILT OFF)
endif()

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(COOLING_CUDA_ARCH_SUMMARY "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(COOLING_CUDA_ARCH_SUMMARY "not set")
endif()

message(STATUS "")
message(STATUS "CoolingSimulation configuration")
message(STATUS "  System:                   ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Processor:                ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C++ compiler:             ${CMAKE_CXX_COMPILER}")
if(COOLING_CUDA_AVAILABLE)
    message(STATUS "  CUDA compiler:            ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "  Build type:               ${CMAKE_BUILD_TYPE}")
message(STATUS "  Serial target built:      ON")
message(STATUS "  HDF5 requested:           ${COOLING_BUILD_HDF5}")
message(STATUS "  HDF5 available:           ${COOLING_HDF5_AVAILABLE}")
message(STATUS "  OpenMP requested:         ${COOLING_BUILD_OPENMP}")
message(STATUS "  OpenMP available:         ${COOLING_OPENMP_AVAILABLE}")
message(STATUS "  OpenMP target built:      ${COOLING_OMP_TARGET_BUILT}")
message(STATUS "  CUDA requested:           ${COOLING_BUILD_CUDA}")
message(STATUS "  CUDA available:           ${COOLING_CUDA_AVAILABLE}")
message(STATUS "  CUDA target built:        ${COOLING_CUDA_TARGET_BUILT}")
message(STATUS "  Native host architecture: ${COOLING_NATIVE_ARCH}")
message(STATUS "  CUDA architectures:       ${COOLING_CUDA_ARCH_SUMMARY}")
message(STATUS "")
