cmake_minimum_required(VERSION 3.18...3.30)

project(ParticlesSimulation LANGUAGES CXX)

# ============================================================
# Options
# ============================================================

set(PARTICLES_BUILD_OPENMP "OFF" CACHE STRING
    "Build OpenMP executable: OFF, ON, or AUTO"
)
set(PARTICLES_BUILD_CUDA "OFF" CACHE STRING
    "Build CUDA executable: OFF, ON, or AUTO"
)
set(PARTICLES_CUDA_ARCHITECTURES "AUTO" CACHE STRING
    "CUDA architectures: AUTO, native, 80, 90, or list like 70;80;90"
)

option(PARTICLES_NATIVE_ARCH
    "Enable native host architecture flags"
    OFF
)

option(PARTICLES_FAST_MATH_CUDA
    "Enable --use_fast_math for CUDA code"
    OFF
)

option(PARTICLES_STRICT_CUDA
    "Fail if CUDA is requested but unavailable"
    OFF
)

set_property(CACHE PARTICLES_BUILD_OPENMP PROPERTY STRINGS OFF ON AUTO)
set_property(CACHE PARTICLES_BUILD_CUDA PROPERTY STRINGS OFF ON AUTO)

# ============================================================
# Validate options
# ============================================================

if(NOT PARTICLES_BUILD_OPENMP MATCHES "^(OFF|ON|AUTO)$")
    message(FATAL_ERROR "PARTICLES_BUILD_OPENMP must be OFF, ON, or AUTO")
endif()

if(NOT PARTICLES_BUILD_CUDA MATCHES "^(OFF|ON|AUTO)$")
    message(FATAL_ERROR "PARTICLES_BUILD_CUDA must be OFF, ON, or AUTO")
endif()

if(APPLE AND NOT PARTICLES_BUILD_CUDA STREQUAL "OFF")
    message(WARNING
        "macOS detected. CUDA is not supported on native Apple Silicon/macOS. "
        "Forcing PARTICLES_BUILD_CUDA=OFF."
    )
    set(PARTICLES_BUILD_CUDA "OFF" CACHE STRING "" FORCE)
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
# Dependencies
# ============================================================

find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Optional OpenMP
set(PARTICLES_OPENMP_AVAILABLE OFF)
if(NOT PARTICLES_BUILD_OPENMP STREQUAL "OFF")
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        set(PARTICLES_OPENMP_AVAILABLE ON)
    elseif(PARTICLES_BUILD_OPENMP STREQUAL "ON")
        message(FATAL_ERROR
            "OpenMP was requested, but OpenMP was not found."
        )
    else()
        message(WARNING
            "OpenMP not found. OpenMP target will not be built."
        )
    endif()
endif()

# Optional CUDA
set(PARTICLES_CUDA_AVAILABLE OFF)
if(NOT PARTICLES_BUILD_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        if(NOT PARTICLES_CUDA_ARCHITECTURES STREQUAL "AUTO"
           AND NOT PARTICLES_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "${PARTICLES_CUDA_ARCHITECTURES}")
        endif()

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        set(PARTICLES_CUDA_AVAILABLE ON)
    elseif(PARTICLES_BUILD_CUDA STREQUAL "ON" OR PARTICLES_STRICT_CUDA)
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

set(PARTICLES_SERIAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/particles.cpp")
set(PARTICLES_OMP_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/particles_omp.cpp")
set(PARTICLES_CUDA_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/particles.cu")

if(NOT EXISTS "${PARTICLES_SERIAL_SRC}")
    message(FATAL_ERROR "Serial source file not found: ${PARTICLES_SERIAL_SRC}")
endif()

# ============================================================
# Helper functions
# ============================================================

function(particles_apply_hdf5 target_name)
    if(TARGET hdf5::hdf5_cpp)
        target_link_libraries(${target_name} PRIVATE hdf5::hdf5_cpp)
        if(TARGET hdf5::hdf5)
            target_link_libraries(${target_name} PRIVATE hdf5::hdf5)
        endif()
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

function(particles_apply_warnings target_name)
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

function(particles_apply_native_arch target_name)
    if(PARTICLES_NATIVE_ARCH AND NOT MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
        )
    endif()
endfunction()

function(particles_apply_common_settings target_name)
    particles_apply_hdf5(${target_name})
    particles_apply_warnings(${target_name})
    particles_apply_native_arch(${target_name})
endfunction()

function(particles_optional_source target_name source_path mode label)
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
    particles_apply_common_settings(${target_name})
endfunction()

# ============================================================
# Targets
# ============================================================

# Serial target: always built
add_executable(particles_serial "${PARTICLES_SERIAL_SRC}")
target_compile_features(particles_serial PRIVATE cxx_std_17)
particles_apply_common_settings(particles_serial)
target_compile_definitions(particles_serial PRIVATE PARTICLES_SERIAL=1)

# OpenMP target: optional
if(PARTICLES_OPENMP_AVAILABLE)
    particles_optional_source(particles_omp "${PARTICLES_OMP_SRC}" "${PARTICLES_BUILD_OPENMP}" "OpenMP")
    if(TARGET particles_omp)
        target_link_libraries(particles_omp PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(particles_omp PRIVATE PARTICLES_OPENMP=1)
    endif()
endif()

# CUDA target: optional
if(PARTICLES_CUDA_AVAILABLE)
    particles_optional_source(particles_cuda "${PARTICLES_CUDA_SRC}" "${PARTICLES_BUILD_CUDA}" "CUDA")
    if(TARGET particles_cuda)
        set_target_properties(particles_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        if(PARTICLES_FAST_MATH_CUDA)
            target_compile_options(particles_cuda PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            )
        endif()

        target_compile_definitions(particles_cuda PRIVATE PARTICLES_CUDA=1)
    endif()
endif()

# ============================================================
# Install
# ============================================================

install(TARGETS particles_serial RUNTIME DESTINATION bin)

if(TARGET particles_omp)
    install(TARGETS particles_omp RUNTIME DESTINATION bin)
endif()

if(TARGET particles_cuda)
    install(TARGETS particles_cuda RUNTIME DESTINATION bin)
endif()

# ============================================================
# Summary
# ============================================================

message(STATUS "")
message(STATUS "ParticlesSimulation configuration")
message(STATUS "  System:                   ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Processor:                ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C++ compiler:             ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build type:               ${CMAKE_BUILD_TYPE}")
message(STATUS "  Serial target:            ON")
message(STATUS "  OpenMP requested:         ${PARTICLES_BUILD_OPENMP}")
message(STATUS "  OpenMP available:         ${PARTICLES_OPENMP_AVAILABLE}")
message(STATUS "  CUDA requested:           ${PARTICLES_BUILD_CUDA}")
message(STATUS "  CUDA available:           ${PARTICLES_CUDA_AVAILABLE}")
message(STATUS "  Native host architecture: ${PARTICLES_NATIVE_ARCH}")
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(PARTICLES_CUDA_ARCH_SUMMARY "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(PARTICLES_CUDA_ARCH_SUMMARY "not set")
endif()
message(STATUS "  CUDA architectures:       ${PARTICLES_CUDA_ARCH_SUMMARY}")
message(STATUS "")
