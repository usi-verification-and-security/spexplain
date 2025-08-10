[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# SpEXplAIn

Copyright 2024 - present Tomáš Kolárik <tomaqa@gmail.com>  
Copyright 2024 Martin Blicha <martin.blicha@gmail.com>  

SpEXplAIn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

SpEXplAIn is a tool that automatically generates Space Explanations of neural network classification.
It gives provable guarantees of the behavior of the network in continuous areas of the
input feature space.
The tool leverages [OpenSMT2](https://github.com/usi-verification-and-security/opensmt) solver compatible
with a range of flexible Craig interpolation algorithms and unsatisfiable core generation.

## Building from source

To build the system from the source code repository, you need a C++20
compliant compiler and the following libraries and headers installed:

 - gmp

SpEXplAIn uses CMake as a build system generator.
We use a wrapper `Makefile` (i.e. GNU Make build system) that allows straightforward building of SpEXplAIn.

To configure and build the project, run the following command inside the SpEXplAIn directory:
```
$ make
```
This will run `cmake -B <build_dir>` and `cmake --build <build_dir>`.
The default `<build_dir>` is `build`, but it can be changed using the command line option `RELEASE_BUILD_DIR`, for example:
`make RELEASE_BUILD_DIR=build-release`.

If the command is not run for the first time, it only rebuilds the sources that are not up-to-date. In the case the `<build_dir>` was removed, it creates it again.

The option `CMAKE_FLAGS` may be used for any additional arguments to be passed to `cmake -B <build_dir>`.
Similarly,
the option `CMAKE_BUILD_FLAGS` may be used for any additional arguments to be passed to `cmake --build <build_dir>`.

### Changing build type

The default build type is Release and `make` is in fact just an alias to `make release`.
In order to build in Debug mode, use
```
$ make debug
```
In this case, options that are related to debug build type should use the `*DEBUG*` variants instead of `*RELEASE*`.
For example `DEBUG_BUILD_DIR=<build_dir>`.
The default `<build_dir>` in debug mode is `build-debug`.

In order to build with an additional verifier [Marabou](https://github.com/NeuralNetworkVerification/Marabou) that is more efficient for simplistic explanations such as abductive or interval explanations than OpenSMT2, run
```
make marabou
```
which will by default use `build-marabou` build directory.
To make the corresponding debug type, use `make marabou-debug`.

In order to build all types, run `make all`.

### Clearing the build

In case one for example needs to rebuild the project from scratch, it can be removed at first:
```
make clean
```
(or `make clean-release`) and then built again with `make` (or `make release`).
In the case of debug mode, one must run `make clean-debug` and `make debug`.
It works the same way with `*-marabou*`.
To remove all, run `make clean-all`.

### Examples

Build with the default values:
```
make
```

Build Debug:
```
make debug
```

Build with Marabou:
```
make marabou
```

## Publications

* Space Explanations of Neural Network Classification, CAV 2025 ([link](https://link.springer.com/chapter/10.1007/978-3-031-98682-6_15))

## Contact
If you have questions, bug reports, or feature requests, please refer to our [GitHub](https://github.com/usi-verification-and-security/spexplain/issues) issue tracker or send us an email.
