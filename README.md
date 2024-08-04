# TORC: Tools for Optimization, Robotics, and Control
This repository is for a currently in-development set of tools for optimization based control of robotic systems.
A full description of the capabilities and documentation of the library will be available soon.

The code is written in C++ with real-time applications in mind. Ultimately, the code will be able to help users 
setup reduced and full order models for robots, and create, manage, and approximate constraints and costs. 

## Dependencies and Supported Systems
Dependencies (required):
- Eigen 3

Dependencies (optional):
- Pinocchio 3.0
- OSQP & OSQP Eigen
- Clarabel.cpp
- IPOPT

Officially Supported Systems:
- Ubuntu 22.04
We hope to expand the supported systems soon.

## CMake Options
A number of external dependency interfaces (optimization solvers, rigid body dynamics) are provided and can be built with the 
library if desired. The following cmake variables can be set to build these interfaces:

- `BUILD_WITH_PINOCCHIO`
- `BUILD_WITH_IPOPT`
- `BUILD_WITH_OSQP`
- `BUILD_WITH_CLARABEL`

We currently default all of these to `ON`. If we want to turn off IPOPT, for example, then we can pass 
`-DBUILD_WITH_IPOPT=OFF`.

### IPOPT
`IPOPT_INC_PATH` and `IPOPT_LIB_PATH` may need to be set. They set the location to search for the header files,
and the location to search for the library (.so) file, respectively. Their default values are:
- `IPOPT_INC_PATH = /usr/local/include/coin-or`
- `IPOPT_LIB_PATH = /usr/local/lib`

These are the default install locations for IPOPT. If you have installed IPOPT in a different location then these
variables must be set accordingly.

### Clarabel
`CLARABEL_INC_PATH` and `CLARABEL_LIB_PATH` may need to be set. They set the location to search for the header files,
and the location of the library (.so) file, respectively. Their default values are:
- `CLARABEL_INC_PATH = ~/Clarabel.cpp/include`
- `CLARABEL_LIB_PATH = ~/Clarabel.cpp/rust_wrapper/target/release/libclarabel_c.so`

If you have installed Clarabel in a different location then these variables must be set accordingly.

### OSQP
If OSQP is installed properly, then the library will be automatically found and linked.

### Pinocchio
If Pinocchio is installed properly, then the library will be automatically found and linked.
Note that we rely on Pinocchio 3.0, which as of 6/18/24 is only available as a build from source 
(but note that it is an officially supported release, not a preview branch).

### CppAD and CppAD Codegen
When installing CppAD, make sure you use the following cmake flags: `cmake -DCMAKE_BUILD_TYPE=Release -Dcppad_cxx_flags=-std=c++17`