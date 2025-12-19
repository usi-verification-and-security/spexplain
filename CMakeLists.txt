cmake_minimum_required(VERSION 3.16)
project(XAI-SMT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_MARABOU "Enable Marabou verifier" OFF)

include(FetchContent)

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
