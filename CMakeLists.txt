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
option(FETCH_PROTOBUF "Download and build protobuf from source instead of using a system install" OFF)

if (NOT FETCH_PROTOBUF)
    find_package(protobuf CONFIG QUIET)
    if (NOT protobuf_FOUND)
        find_package(Protobuf MODULE QUIET)
    endif()
    if (NOT protobuf_FOUND AND NOT Protobuf_FOUND)
        message(FATAL_ERROR
            "Could not find Protobuf.\n"
            "Either point CMake at an install with -DCMAKE_PREFIX_PATH=<prefix>, or build it from\n"
            "source with -DFETCH_PROTOBUF=ON (needs no system packages and no root).")
    endif()
endif()

include(FetchContent)

if (FETCH_PROTOBUF)
    # v3.21.12 is the last release before protobuf took on its abseil dependency, which keeps this
    # build small and self-contained. Its own CMake minimum is modern enough for CMake 4.x.
    set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
    set(protobuf_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(protobuf_INSTALL OFF CACHE INTERNAL "")
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    # We only parse a serialised model; gzip streams are unused, so do not require zlib.
    set(protobuf_WITH_ZLIB OFF CACHE INTERNAL "")
    set(protobuf_BUILD_PROTOC_BINARIES ON CACHE INTERNAL "")

    # Fetched as a source tarball rather than a git clone: CMake downloads it with its own
    # bundled libcurl, which sidesteps environments where `git clone https://` is broken (a
    # frequent case in HPC containers, where host libraries injected via LD_LIBRARY_PATH shadow
    # the container's libcurl/libnghttp2).
    #
    # Offline / restricted-network escape hatches:
    #   -DPROTOBUF_SOURCE_DIR=/path/to/protobuf-21.12   already-extracted sources, no download
    #   -DPROTOBUF_URL=/path/to/protobuf-v21.12.tar.gz  a local tarball, or a mirror URL
    set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.12.tar.gz"
        CACHE STRING "URL or local path of the protobuf source archive")
    set(PROTOBUF_URL_HASH "SHA256=22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de"
        CACHE STRING "Expected hash of PROTOBUF_URL; set to empty to skip the check")
    set(PROTOBUF_SOURCE_DIR "" CACHE PATH "Pre-extracted protobuf sources; skips downloading")

    if (PROTOBUF_SOURCE_DIR)
        message(STATUS "Using protobuf sources from ${PROTOBUF_SOURCE_DIR}")
        FetchContent_Declare(protobuf SOURCE_DIR "${PROTOBUF_SOURCE_DIR}")
    elseif (PROTOBUF_URL_HASH)
        FetchContent_Declare(protobuf URL "${PROTOBUF_URL}" URL_HASH "${PROTOBUF_URL_HASH}")
    else()
        FetchContent_Declare(protobuf URL "${PROTOBUF_URL}")
    endif()
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

option(REGENERATE_ONNX_PROTO "Regenerate onnx.pb.{h,cc} from onnx.proto using the local protoc" OFF)
if (SPEXPLAIN_FETCHED_PROTOBUF AND NOT REGENERATE_ONNX_PROTO)
    # The checked-in gencode is pinned to one exact protobuf release and cannot match a fetched one.
    message(STATUS "Enabling REGENERATE_ONNX_PROTO: the fetched protobuf differs from the pinned gencode")
    set(REGENERATE_ONNX_PROTO ON)
endif()

# Find OpenSMT2
SET(OPENSMT_HOME CACHE STRING "OpenSMT installation directory")
# find_package(OpenSMT 2.6.0 EXACT CONFIG PATHS ${OPENSMT_HOME} NO_DEFAULT_PATH)

# if (OpenSMT_FOUND)
# else(OpenSMT_FOUND)
    # Same escape hatches as protobuf, for environments where `git clone https://` is broken:
    #   -DOPENSMT_SOURCE_DIR=/path/to/opensmt   already-extracted sources, no download
    #   -DOPENSMT_URL=<url-or-local-tarball>    fetch a tarball with CMake's downloader, not git
    # No hash is pinned because the default URL tracks a branch, whose content changes.
    set(OPENSMT_URL "" CACHE STRING "URL or local path of an OpenSMT source archive (branch tarball)")
    set(OPENSMT_SOURCE_DIR "" CACHE PATH "Pre-extracted OpenSMT sources; skips downloading")

    if (OPENSMT_SOURCE_DIR)
        message(STATUS "Using OpenSMT sources from ${OPENSMT_SOURCE_DIR}")
        FetchContent_Declare(OpenSMT SOURCE_DIR "${OPENSMT_SOURCE_DIR}")
    elseif (OPENSMT_URL)
        message(STATUS "Fetching OpenSMT from ${OPENSMT_URL}")
        FetchContent_Declare(OpenSMT URL "${OPENSMT_URL}")
    else()
        FetchContent_Declare(
            OpenSMT
            GIT_REPOSITORY https://github.com/usi-verification-and-security/opensmt.git
            GIT_TAG origin/spexplain
            GIT_SHALLOW true
            GIT_PROGRESS true
        )
    endif()
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
