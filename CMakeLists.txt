cmake_minimum_required(VERSION 3.16)
project(XAI-SMT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_MARABOU "Enable Marabou verifier" OFF)

# macOS ships bison 2.3, which OpenSMT's parser cannot be built with; Homebrew has 3.x.
# This must stay guarded: on Linux the path does not exist, and FORCE-ing it into the cache
# makes the OpenSMT build fail with a missing bison.
if (APPLE)
    list(PREPEND CMAKE_PROGRAM_PATH "/opt/homebrew/bin")
    if (EXISTS "/opt/homebrew/bin/bison")
        set(BISON_EXECUTABLE "/opt/homebrew/bin/bison" CACHE FILEPATH "Bison executable" FORCE)
    endif()
endif()

# Protobuf. Three ways to get it, in order of preference:
#   1. installed on the system         -> found below (CONFIG mode first: protobuf >= 22 ships a
#                                         config package and depends on abseil; MODULE mode is the
#                                         fallback for older installs)
#   2. installed in a custom prefix    -> cmake -DCMAKE_PREFIX_PATH=<prefix>
#   3. not available at all            -> -DFETCH_PROTOBUF=ON builds it from source into the build
#                                         tree, needing nothing installed on the machine. This is
#                                         the option for a fixed container image with no root.
# AUTO (default): use a system protobuf when there is one, otherwise build it from source.
#                 This is what makes a plain `make` work both on a developer machine with
#                 protobuf installed and in a bare container with nothing installed.
# ON            : always build from source, ignoring any system install.
# OFF           : require a system install and fail if it is missing.
# Earlier revisions declared these with option(), i.e. as BOOL. A stale BOOL cache entry would now
# be read with the new meaning (OFF = "forbid building from source"), which is not what it meant
# then, so migrate any such entry back to AUTO instead of failing on an existing build directory.
foreach (_var FETCH_PROTOBUF REGENERATE_ONNX_PROTO)
    get_property(_type CACHE ${_var} PROPERTY TYPE)
    if (_type STREQUAL "BOOL")
        message(STATUS "Migrating stale boolean ${_var} cache entry to AUTO")
        unset(${_var} CACHE)
    endif()
endforeach()

set(FETCH_PROTOBUF "AUTO" CACHE STRING "Build protobuf from source: AUTO, ON or OFF")
set_property(CACHE FETCH_PROTOBUF PROPERTY STRINGS AUTO ON OFF)

if (NOT FETCH_PROTOBUF STREQUAL "ON")
    find_package(protobuf CONFIG QUIET)
    if (NOT protobuf_FOUND)
        find_package(Protobuf MODULE QUIET)
    endif()
endif()

set(SPEXPLAIN_HAVE_SYSTEM_PROTOBUF FALSE)
if (protobuf_FOUND OR Protobuf_FOUND)
    set(SPEXPLAIN_HAVE_SYSTEM_PROTOBUF TRUE)
endif()

if (FETCH_PROTOBUF STREQUAL "OFF" AND NOT SPEXPLAIN_HAVE_SYSTEM_PROTOBUF)
    message(FATAL_ERROR
        "Could not find Protobuf, and FETCH_PROTOBUF=OFF forbids building it.\n"
        "Point CMake at an install with -DCMAKE_PREFIX_PATH=<prefix>, or drop FETCH_PROTOBUF=OFF\n"
        "to let it be built from source automatically.")
endif()

include(FetchContent)

if (FETCH_PROTOBUF STREQUAL "ON")
    message(STATUS "Building protobuf from source (FETCH_PROTOBUF=ON; any system install is ignored)")
elseif (SPEXPLAIN_HAVE_SYSTEM_PROTOBUF)
    message(STATUS "Using the system protobuf")
else()
    message(STATUS "No system protobuf found; building it from source")
endif()

if (NOT SPEXPLAIN_HAVE_SYSTEM_PROTOBUF OR FETCH_PROTOBUF STREQUAL "ON")
    # v3.21.12 is the last release before protobuf took on its abseil dependency, which keeps this
    # build small and self-contained. Its own CMake minimum is modern enough for CMake 4.x.
    set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
    set(protobuf_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(protobuf_INSTALL OFF CACHE INTERNAL "")
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    # We only parse a serialised model; gzip streams are unused, so do not require zlib.
    set(protobuf_WITH_ZLIB OFF CACHE INTERNAL "")
    set(protobuf_BUILD_PROTOC_BINARIES ON CACHE INTERNAL "")

    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v3.21.12
        GIT_SHALLOW true
        GIT_PROGRESS true
    )
    message(STATUS "Building protobuf from source (FETCH_PROTOBUF=ON); this takes a few minutes")
    FetchContent_MakeAvailable(protobuf)
    set(SPEXPLAIN_FETCHED_PROTOBUF ON)
endif()

# Abseil is only needed because protobuf >= 22 depends on it; nothing in this project includes
# absl headers itself. Distro protobuf 3.x and the fetched v3.21.12 both predate that dependency,
# so do not hard-require abseil -- link it only when it is actually present.
if (SPEXPLAIN_FETCHED_PROTOBUF)
    # The fetched v3.21.12 predates the abseil dependency; a system abseil found here would be
    # linked in for nothing and could even mismatch, so do not look for it at all.
    message(STATUS "Skipping abseil: the fetched protobuf does not need it")
else()
    find_package(absl CONFIG QUIET)
    if (absl_FOUND)
        message(STATUS "Abseil found; linking it alongside protobuf")
    else()
        message(STATUS "Abseil not found; assuming a protobuf that does not require it")
    endif()
endif()

# The checked-in src/onnx-1.15.0/onnx.pb.{h,cc} carry `#if PROTOBUF_VERSION != <n>` + `#error`,
# so they compile against exactly one protobuf release and break against any other. Regenerating
# them from onnx.proto with whatever protoc is at hand removes that coupling entirely, which is why
# AUTO (the default) regenerates whenever a protoc is available.
set(REGENERATE_ONNX_PROTO "AUTO" CACHE STRING
    "Regenerate onnx.pb.{h,cc} from onnx.proto: AUTO (whenever protoc is available), ON or OFF")
set_property(CACHE REGENERATE_ONNX_PROTO PROPERTY STRINGS AUTO ON OFF)

if (TARGET protobuf::protoc)
    set(SPEXPLAIN_PROTOC $<TARGET_FILE:protobuf::protoc>)
elseif (Protobuf_PROTOC_EXECUTABLE)
    set(SPEXPLAIN_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
else()
    set(SPEXPLAIN_PROTOC "")
endif()

if (REGENERATE_ONNX_PROTO STREQUAL "OFF")
    set(SPEXPLAIN_REGENERATE_ONNX FALSE)
elseif (REGENERATE_ONNX_PROTO STREQUAL "ON")
    if (NOT SPEXPLAIN_PROTOC)
        message(FATAL_ERROR "REGENERATE_ONNX_PROTO=ON but no protoc was found alongside protobuf")
    endif()
    set(SPEXPLAIN_REGENERATE_ONNX TRUE)
else()
    if (SPEXPLAIN_PROTOC)
        set(SPEXPLAIN_REGENERATE_ONNX TRUE)
    else()
        set(SPEXPLAIN_REGENERATE_ONNX FALSE)
        if (SPEXPLAIN_FETCHED_PROTOBUF)
            message(FATAL_ERROR "Built protobuf from source but protoc is missing; cannot continue")
        endif()
        message(WARNING
            "No protoc found: falling back to the checked-in onnx.pb.{h,cc}, which only compile "
            "against one exact protobuf version and will fail if yours differs.")
    endif()
endif()

if (SPEXPLAIN_REGENERATE_ONNX)
    message(STATUS "onnx.pb.{h,cc} will be regenerated with the local protoc")
else()
    message(STATUS "Using the checked-in onnx.pb.{h,cc}")
endif()

# Find OpenSMT2
SET(OPENSMT_HOME CACHE STRING "OpenSMT installation directory")
# find_package(OpenSMT 2.6.0 EXACT CONFIG PATHS ${OPENSMT_HOME} NO_DEFAULT_PATH)

# if (OpenSMT_FOUND)
# else(OpenSMT_FOUND)
    FetchContent_Declare(
        OpenSMT
        GIT_REPOSITORY https://github.com/usi-verification-and-security/opensmt.git
        GIT_TAG origin/spexplain
        GIT_SHALLOW true
        GIT_PROGRESS true
    )
    set(BUILD_EXECUTABLES OFF CACHE INTERNAL "")
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(BUILD_STATIC_LIBS ON CACHE INTERNAL "")
    set(PACKAGE_TESTS OFF CACHE INTERNAL "")
    message("Getting OpenSMT...")
    FetchContent_MakeAvailable(OpenSMT)
    message("OpenSMT downloaded")
    add_library(OpenSMT::OpenSMT ALIAS OpenSMT-static)
# endif(OpenSMT_FOUND)

# if (NOT OpenSMT_FOUND)
    # target_compile_definitions(XAI-SMT PUBLIC OPENSMT_LOCAL_BUILD)
    #message("Directories:${opensmt_SOURCE_DIR}; ${opensmt_BINARY_DIR}")
    target_include_directories(OpenSMT-static PUBLIC
    ${opensmt_SOURCE_DIR}/src
    )
# endif()


if (ENABLE_MARABOU)
    FetchContent_Declare(
            Marabou
            # GIT_REPOSITORY https://github.com/NeuralNetworkVerification/Marabou.git
            # GIT_TAG 215828c64e624be7917e69e4e873c746d8df25a2
            GIT_REPOSITORY https://github.com/Tomaqa/Marabou.git
            GIT_TAG origin/spexplain
            GIT_SHALLOW true
            GIT_PROGRESS true
        )
    FetchContent_MakeAvailable(Marabou)
    add_library(Marabou::Marabou ALIAS MarabouHelper)
    get_target_property(MARABOU_INCLUDE_DIRS MarabouHelper INCLUDE_DIRECTORIES)

    add_definitions(-DMARABOU)
endif()

set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

include_directories(${SOURCE_DIR})

add_subdirectory(${SOURCE_DIR})
