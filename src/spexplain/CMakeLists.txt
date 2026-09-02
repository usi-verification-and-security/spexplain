add_library(spexplain OBJECT
        network/Network2.h
        network/Network2.cpp)

# onnx.pb.{h,cc}: either the checked-in copies (pinned to one exact protobuf version by an
# `#if PROTOBUF_VERSION != ...` / `#error` guard) or freshly generated ones. Generated output goes
# to <build>/generated/onnx-1.15.0/ so the existing `#include <onnx-1.15.0/onnx.pb.h>` keeps
# resolving. That directory must be added with BEFORE and *without* SYSTEM: system include paths
# are searched after normal ones, so marking it SYSTEM would let the stale checked-in header under
# ${SOURCE_DIR} win and the build would fail with a protobuf version mismatch.
set(ONNX_PROTO_DIR "${PROJECT_SOURCE_DIR}/src/onnx-1.15.0")
if (REGENERATE_ONNX_PROTO)
    if (TARGET protobuf::protoc)
        set(_protoc $<TARGET_FILE:protobuf::protoc>)
    elseif (Protobuf_PROTOC_EXECUTABLE)
        set(_protoc "${Protobuf_PROTOC_EXECUTABLE}")
    else()
        message(FATAL_ERROR "REGENERATE_ONNX_PROTO=ON but no protoc was found alongside protobuf")
    endif()

    set(ONNX_PB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated")
    set(ONNX_PB_OUT_DIR "${ONNX_PB_INCLUDE_DIR}/onnx-1.15.0")
    file(MAKE_DIRECTORY "${ONNX_PB_OUT_DIR}")
    add_custom_command(
        OUTPUT "${ONNX_PB_OUT_DIR}/onnx.pb.cc" "${ONNX_PB_OUT_DIR}/onnx.pb.h"
        COMMAND ${_protoc} -I "${ONNX_PROTO_DIR}" --cpp_out "${ONNX_PB_OUT_DIR}" onnx.proto
        DEPENDS "${ONNX_PROTO_DIR}/onnx.proto"
        COMMENT "Regenerating onnx.pb.{h,cc} with the local protoc"
        VERBATIM)
    set(ONNX_PB_SOURCE "${ONNX_PB_OUT_DIR}/onnx.pb.cc")
else()
    set(ONNX_PB_SOURCE "${ONNX_PROTO_DIR}/onnx.pb.cc")
endif()

target_sources(spexplain
PRIVATE
    common/Bound.cpp
    common/Interval.cpp
    common/MString.cpp
    common/Print.cpp
    common/TensorUtils.cpp
    network/Network.cpp
    ${ONNX_PB_SOURCE}
    network/OnnxParser.cpp
    network/FCLayer.cpp
    network/ReLULayer.cpp
    network/SigmoidLayer.cpp
    network/CNNLayer.cpp
    network/TransposeLayer.cpp
    network/MaxPoolLayer.cpp
    network/AddLayer.cpp
    network/Dataset.cpp
    framework/Config.cpp
    framework/Framework.cpp
    framework/Parse.cpp
    framework/Preprocess.cpp
    framework/Print.cpp
    framework/Utils.cpp
    framework/expand/Expand.cpp
    framework/expand/strategy/Factory.cpp
    framework/expand/strategy/Strategy.cpp
    framework/expand/strategy/AbductiveStrategy.cpp
    framework/expand/strategy/TrialAndErrorStrategy.cpp
    framework/expand/strategy/UnsatCoreStrategy.cpp
    framework/expand/strategy/SliceStrategy.cpp
    framework/expand/strategy/opensmt/Strategy.cpp
    framework/expand/strategy/opensmt/UnsatCoreStrategy.cpp
    framework/expand/strategy/opensmt/InterpolationStrategy.cpp
    framework/explanation/ConjunctExplanation.cpp
    framework/explanation/Explanation.cpp
    framework/explanation/IntervalExplanation.cpp
    framework/explanation/PartialExplanation.cpp
    framework/explanation/VarBound.cpp
    framework/explanation/opensmt/FormulaExplanation.cpp
PUBLIC
)

if (REGENERATE_ONNX_PROTO)
    target_include_directories(spexplain BEFORE PUBLIC ${ONNX_PB_INCLUDE_DIR})
endif()

target_include_directories(spexplain SYSTEM PUBLIC
    ${opensmt_SOURCE_DIR}/src
    ${GMP_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

if (ENABLE_MARABOU)
    target_include_directories(spexplain SYSTEM PUBLIC
        ${MARABOU_INCLUDE_DIRS}
    )
endif()

target_compile_options(spexplain
PRIVATE
    -Wall -Wextra -pedantic -Wshadow
)

target_link_libraries(spexplain
PUBLIC
    protobuf::libprotobuf
)

if (absl_FOUND)
    target_link_libraries(spexplain
    PUBLIC
        absl::log
        absl::check
        absl::base
        absl::strings
    )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_options(spexplain
PRIVATE
    -Weffc++ -Wno-non-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wmissing-declarations
)
endif()


add_executable(SpEXplAIn-bin
    bin/main.cpp
    ${SOURCE_DIR}/verifiers/Verifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier2.cpp
)

if (ENABLE_MARABOU)
    target_sources(SpEXplAIn-bin PUBLIC
        ${SOURCE_DIR}/verifiers/marabou/MarabouVerifier.cpp
    )
endif()

set_target_properties(SpEXplAIn-bin
PROPERTIES
    OUTPUT_NAME spexplain
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(SpEXplAIn-bin PUBLIC
    spexplain
    OpenSMT::OpenSMT
)

if (ENABLE_MARABOU)
    target_link_libraries(SpEXplAIn-bin PUBLIC
        MarabouHelper
    )
endif()

add_executable(onnx-eval
    bin/onnx_eval.cpp
    ${SOURCE_DIR}/verifiers/Verifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier2.cpp
)

if (ENABLE_MARABOU)
    target_sources(onnx-eval PUBLIC
        ${SOURCE_DIR}/verifiers/marabou/MarabouVerifier.cpp
    )
endif()

set_target_properties(onnx-eval
PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(onnx-eval PUBLIC
    spexplain
    OpenSMT::OpenSMT
)

if (ENABLE_MARABOU)
    target_link_libraries(onnx-eval PUBLIC
        MarabouHelper
    )
endif()

add_executable(compare-nnet-onnx
    bin/compare_nnet_onnx.cpp
    ${SOURCE_DIR}/verifiers/Verifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier2.cpp
)

if (ENABLE_MARABOU)
    target_sources(compare-nnet-onnx PUBLIC
        ${SOURCE_DIR}/verifiers/marabou/MarabouVerifier.cpp
    )
endif()

set_target_properties(compare-nnet-onnx
PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(compare-nnet-onnx PUBLIC
    spexplain
    OpenSMT::OpenSMT
)

if (ENABLE_MARABOU)
    target_link_libraries(compare-nnet-onnx PUBLIC
        MarabouHelper
    )
endif()

