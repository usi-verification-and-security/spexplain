add_library(xspace OBJECT)

target_sources(xspace
PRIVATE
    common/Bound.cpp
    common/Interval.cpp
    common/Print.cpp
    network/Network.cpp
    network/Dataset.cpp
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

target_include_directories(xspace SYSTEM PUBLIC
    ${opensmt_SOURCE_DIR}/src
    ${GMP_INCLUDE_DIR}
)

if (ENABLE_MARABOU)
    target_include_directories(xspace SYSTEM PUBLIC
        ${MARABOU_INCLUDE_DIRS}
    )
endif()

target_compile_options(xspace
PRIVATE
    -Wall -Wextra -pedantic -Wshadow
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
target_compile_options(xspace
PRIVATE
    -Weffc++ -Wno-non-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wmissing-declarations
)
endif()

add_executable(SpEXplAIn-bin
    bin/main.cpp
    ${SOURCE_DIR}/verifiers/opensmt/OpenSMTVerifier.cpp
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
    xspace
    OpenSMT::OpenSMT
)

if (ENABLE_MARABOU)
    target_link_libraries(SpEXplAIn-bin PUBLIC
        MarabouHelper
    )
endif()
