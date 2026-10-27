### This is just a wrapper Makefile script on top of CMake

BUILD_DIR_BASE = build

RELEASE_BUILD_DIR_SUFFIX =
DEBUG_BUILD_DIR_SUFFIX = -debug
## These can be customized via command line
RELEASE_BUILD_DIR = $(BUILD_DIR_BASE)$(RELEASE_BUILD_DIR_SUFFIX)
DEBUG_BUILD_DIR = $(BUILD_DIR_BASE)$(DEBUG_BUILD_DIR_SUFFIX)

MARABOU_BUILD_DIR_SUFFIX = -marabou
## These can be customized via command line
MARABOU_RELEASE_BUILD_DIR = $(BUILD_DIR_BASE)$(MARABOU_BUILD_DIR_SUFFIX)$(RELEASE_BUILD_DIR_SUFFIX)
MARABOU_DEBUG_BUILD_DIR = $(BUILD_DIR_BASE)$(MARABOU_BUILD_DIR_SUFFIX)$(DEBUG_BUILD_DIR_SUFFIX)

ONNX_BUILD_DIR_SUFFIX = -onnx
## These can be customized via command line
ONNX_RELEASE_BUILD_DIR = $(BUILD_DIR_BASE)$(ONNX_BUILD_DIR_SUFFIX)$(RELEASE_BUILD_DIR_SUFFIX)
ONNX_DEBUG_BUILD_DIR = $(BUILD_DIR_BASE)$(ONNX_BUILD_DIR_SUFFIX)$(DEBUG_BUILD_DIR_SUFFIX)

## The engine for ONNX networks is built separately from a frozen git revision,
## which reproduces the previous experiments; its sources are extracted into <build_dir>/source
## This can be customized via command line
ONNX_ENGINE_REV = onnx-engine-v1
## The name of the target in the frozen revision
ONNX_ENGINE_TARGET = SpEXplAIn-bin
ONNX_ENGINE_EXECUTABLE = spexplain

DEFAULT_INSTALL_DIR := /usr/local
USER_INSTALL_DIR := $(INSTALL_DIR)
## This can be customized via command line,
## it takes effect within both `cmake --build` and `cmake --install`
INSTALL_DIR = $(DEFAULT_INSTALL_DIR)

## These can be customized via command line
## Arguments for `cmake -B <build_dir>`
CMAKE_FLAGS += -DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR)

## These can be customized via command line
## Git revisions (branches, tags or commits) of the dependencies (default: see CMakeLists.txt)
ifneq ($(OPENSMT_GIT_TAG),)
  CMAKE_FLAGS += -DOPENSMT_GIT_TAG=$(OPENSMT_GIT_TAG)
endif
ifneq ($(MARABOU_GIT_TAG),)
  CMAKE_FLAGS += -DMARABOU_GIT_TAG=$(MARABOU_GIT_TAG)
endif

RELEASE_ONLY_CMAKE_FLAGS +=
DEBUG_ONLY_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Debug
RELEASE_CMAKE_FLAGS += $(CMAKE_FLAGS) $(RELEASE_ONLY_CMAKE_FLAGS)
DEBUG_CMAKE_FLAGS += $(CMAKE_FLAGS) $(DEBUG_ONLY_CMAKE_FLAGS)

MARABOU_ONLY_CMAKE_FLAGS += -DENABLE_MARABOU=ON
MARABOU_RELEASE_CMAKE_FLAGS += $(CMAKE_FLAGS) $(RELEASE_ONLY_CMAKE_FLAGS) $(MARABOU_ONLY_CMAKE_FLAGS)
MARABOU_DEBUG_CMAKE_FLAGS += $(CMAKE_FLAGS) $(DEBUG_ONLY_CMAKE_FLAGS) $(MARABOU_ONLY_CMAKE_FLAGS)

ONNX_ONLY_CMAKE_FLAGS +=
ONNX_RELEASE_CMAKE_FLAGS += $(CMAKE_FLAGS) $(RELEASE_ONLY_CMAKE_FLAGS) $(ONNX_ONLY_CMAKE_FLAGS)
ONNX_DEBUG_CMAKE_FLAGS += $(CMAKE_FLAGS) $(DEBUG_ONLY_CMAKE_FLAGS) $(ONNX_ONLY_CMAKE_FLAGS)

## These can be customized via command line
## Arguments for `cmake --build <build_dir>`
CMAKE_BUILD_FLAGS +=

RELEASE_ONLY_CMAKE_BUILD_FLAGS +=
DEBUG_ONLY_CMAKE_BUILD_FLAGS +=
RELEASE_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(RELEASE_ONLY_CMAKE_BUILD_FLAGS)
DEBUG_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(DEBUG_ONLY_CMAKE_BUILD_FLAGS)

MARABOU_ONLY_CMAKE_BUILD_FLAGS +=
MARABOU_RELEASE_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(RELEASE_ONLY_CMAKE_BUILD_FLAGS) $(MARABOU_ONLY_CMAKE_BUILD_FLAGS)
MARABOU_DEBUG_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(DEBUG_ONLY_CMAKE_BUILD_FLAGS) $(MARABOU_ONLY_CMAKE_BUILD_FLAGS)

ONNX_ONLY_CMAKE_BUILD_FLAGS += --target $(ONNX_ENGINE_TARGET)
ONNX_RELEASE_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(RELEASE_ONLY_CMAKE_BUILD_FLAGS) $(ONNX_ONLY_CMAKE_BUILD_FLAGS)
ONNX_DEBUG_CMAKE_BUILD_FLAGS += $(CMAKE_BUILD_FLAGS) $(DEBUG_ONLY_CMAKE_BUILD_FLAGS) $(ONNX_ONLY_CMAKE_BUILD_FLAGS)

## These can be customized via command line
## Arguments for `cmake --install <build_dir>`
CMAKE_INSTALL_FLAGS +=
ifneq ($(USER_INSTALL_DIR),)
  CMAKE_INSTALL_FLAGS += --prefix=$(USER_INSTALL_DIR)
endif

RELEASE_ONLY_CMAKE_INSTALL_FLAGS +=
DEBUG_ONLY_CMAKE_INSTALL_FLAGS +=
RELEASE_CMAKE_INSTALL_FLAGS += $(CMAKE_INSTALL_FLAGS) $(RELEASE_ONLY_CMAKE_INSTALL_FLAGS)
DEBUG_CMAKE_INSTALL_FLAGS += $(CMAKE_INSTALL_FLAGS) $(DEBUG_ONLY_CMAKE_INSTALL_FLAGS)

MARABOU_ONLY_CMAKE_INSTALL_FLAGS +=
MARABOU_RELEASE_CMAKE_INSTALL_FLAGS += $(CMAKE_INSTALL_FLAGS) $(RELEASE_ONLY_CMAKE_INSTALL_FLAGS) $(MARABOU_ONLY_CMAKE_INSTALL_FLAGS)
MARABOU_DEBUG_CMAKE_INSTALL_FLAGS += $(CMAKE_INSTALL_FLAGS) $(DEBUG_ONLY_CMAKE_INSTALL_FLAGS) $(MARABOU_ONLY_CMAKE_INSTALL_FLAGS)

################################################################

.PHONY: default all

default: release

all: release debug marabou-release marabou-debug

################################

.PHONY: release debug
.PHONY: _dir-release _dir-debug _build-release _build-debug

release: _dir-release _build-release

_dir-release: $(RELEASE_BUILD_DIR)

_build-release: _dir-release
	cmake --build $(RELEASE_BUILD_DIR) $(RELEASE_CMAKE_BUILD_FLAGS)

$(RELEASE_BUILD_DIR):
	cmake -B $(RELEASE_BUILD_DIR) $(RELEASE_CMAKE_FLAGS)

debug: _dir-debug _build-debug

_dir-debug: $(DEBUG_BUILD_DIR)

_build-debug: _dir-debug
	cmake --build $(DEBUG_BUILD_DIR) $(DEBUG_CMAKE_BUILD_FLAGS)

$(DEBUG_BUILD_DIR):
	cmake -B $(DEBUG_BUILD_DIR) $(DEBUG_CMAKE_FLAGS)

################################

.PHONY: marabou marabou-release marabou-debug
.PHONY: _dir-marabou-release _dir-marabou-debug _build-marabou-release _build-marabou-debug

marabou: marabou-release

marabou-release: _dir-marabou-release _build-marabou-release

_dir-marabou-release: $(MARABOU_RELEASE_BUILD_DIR)

_build-marabou-release: _dir-marabou-release
	cmake --build $(MARABOU_RELEASE_BUILD_DIR) $(MARABOU_RELEASE_CMAKE_BUILD_FLAGS)

$(MARABOU_RELEASE_BUILD_DIR):
	cmake -B $(MARABOU_RELEASE_BUILD_DIR) $(MARABOU_RELEASE_CMAKE_FLAGS)

marabou-debug: _dir-marabou-debug _build-marabou-debug

_dir-marabou-debug: $(MARABOU_DEBUG_BUILD_DIR)

_build-marabou-debug: _dir-marabou-debug
	cmake --build $(MARABOU_DEBUG_BUILD_DIR) $(MARABOU_DEBUG_CMAKE_BUILD_FLAGS)

$(MARABOU_DEBUG_BUILD_DIR):
	cmake -B $(MARABOU_DEBUG_BUILD_DIR) $(MARABOU_DEBUG_CMAKE_FLAGS)

################################

.PHONY: onnx onnx-release onnx-debug
.PHONY: _dir-onnx-release _dir-onnx-debug _build-onnx-release _build-onnx-debug

onnx: onnx-release

onnx-release: _dir-onnx-release _build-onnx-release

_dir-onnx-release: $(ONNX_RELEASE_BUILD_DIR)

_build-onnx-release: _dir-onnx-release
	cmake --build $(ONNX_RELEASE_BUILD_DIR) $(ONNX_RELEASE_CMAKE_BUILD_FLAGS)
	ln -sfn $(ONNX_ENGINE_EXECUTABLE) $(ONNX_RELEASE_BUILD_DIR)/engine

$(ONNX_RELEASE_BUILD_DIR):
	mkdir -p $(ONNX_RELEASE_BUILD_DIR)/source
	git archive $(ONNX_ENGINE_REV) | tar -x -C $(ONNX_RELEASE_BUILD_DIR)/source
	cmake -S $(ONNX_RELEASE_BUILD_DIR)/source -B $(ONNX_RELEASE_BUILD_DIR) $(ONNX_RELEASE_CMAKE_FLAGS)

onnx-debug: _dir-onnx-debug _build-onnx-debug

_dir-onnx-debug: $(ONNX_DEBUG_BUILD_DIR)

_build-onnx-debug: _dir-onnx-debug
	cmake --build $(ONNX_DEBUG_BUILD_DIR) $(ONNX_DEBUG_CMAKE_BUILD_FLAGS)
	ln -sfn $(ONNX_ENGINE_EXECUTABLE) $(ONNX_DEBUG_BUILD_DIR)/engine

$(ONNX_DEBUG_BUILD_DIR):
	mkdir -p $(ONNX_DEBUG_BUILD_DIR)/source
	git archive $(ONNX_ENGINE_REV) | tar -x -C $(ONNX_DEBUG_BUILD_DIR)/source
	cmake -S $(ONNX_DEBUG_BUILD_DIR)/source -B $(ONNX_DEBUG_BUILD_DIR) $(ONNX_DEBUG_CMAKE_FLAGS)

################################
################################

##++ Tests are not implemented yet

# .PHONY: test test-all

# test: test-release

# test-all: test-release test-debug test-marabou-release test-marabou-debug

################################

# .PHONY: test-release test-debug

# test-release:
# 	ctest --test-dir $(RELEASE_BUILD_DIR)

# test-debug:
# 	ctest --test-dir $(DEBUG_BUILD_DIR)

################################

# .PHONY: test-marabou test-marabou-release test-marabou-debug

# test-marabou: test-marabou-release

# test-marabou-release:
# 	ctest --test-dir $(MARABOU_RELEASE_BUILD_DIR)

# test-marabou-debug:
# 	ctest --test-dir $(MARABOU_DEBUG_BUILD_DIR)

################################
################################

.PHONY: clean clean-all

clean: clean-release

clean-all: clean-release clean-debug clean-marabou-release clean-marabou-debug

################################

.PHONY: clean-release clean-debug

clean-release:
	rm -fr $(RELEASE_BUILD_DIR)

clean-debug:
	rm -fr $(DEBUG_BUILD_DIR)

################################

.PHONY: clean-marabou clean-marabou-release clean-marabou-debug

clean-marabou: clean-marabou-release

clean-marabou-release:
	rm -fr $(MARABOU_RELEASE_BUILD_DIR)

clean-marabou-debug:
	rm -fr $(MARABOU_DEBUG_BUILD_DIR)

################################

.PHONY: clean-onnx clean-onnx-release clean-onnx-debug

clean-onnx: clean-onnx-release

clean-onnx-release:
	rm -fr $(ONNX_RELEASE_BUILD_DIR)

clean-onnx-debug:
	rm -fr $(ONNX_DEBUG_BUILD_DIR)

################################
################################

.PHONY: install

install: install-release

################################

.PHONY: install-release install-debug

install-release: release
	cmake --install $(RELEASE_BUILD_DIR) $(RELEASE_CMAKE_INSTALL_FLAGS)

install-debug: debug
	cmake --install $(DEBUG_BUILD_DIR) $(DEBUG_CMAKE_INSTALL_FLAGS)

################################

.PHONY: install-marabou install-marabou-release install-marabou-debug

install-marabou: install-marabou-release

install-marabou-release: marabou-release
	cmake --install $(MARABOU_RELEASE_BUILD_DIR) $(MARABOU_RELEASE_CMAKE_INSTALL_FLAGS)

install-marabou-debug: MARABOU-debug
	cmake --install $(MARABOU_DEBUG_BUILD_DIR) $(MARABOU_DEBUG_CMAKE_INSTALL_FLAGS)
