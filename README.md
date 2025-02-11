# TORC: Tools for Optimization, Robotics, and Control
This repository is for a currently in-development set of tools for optimization based control of robotic systems.
A full description of the capabilities and documentation of the library will be available soon.

The code is written in C++ with real-time applications in mind. Ultimately, the code will be able to help users 
setup full order models for robots, and create, manage, and approximate constraints and costs. 

## Dependencies and Supported Systems
Dependencies:
- Eigen 3
- Pinocchio 3.0
- OSQP & OSQP++

Officially Supported Systems:
- Ubuntu 22.04
We hope to expand the supported systems soon.

If you have installed Clarabel in a different location then these variables must be set accordingly.

## OSQP
If OSQP is installed properly, then the library will be automatically found and linked.

## Pinocchio
If Pinocchio is installed properly, then the library will be automatically found and linked.
Note that we rely on Pinocchio 3.0, which as of 6/18/24 is only available as a build from source 
(but note that it is an officially supported release, not a preview branch).

## CppAD and CppAD Codegen
When installing CppAD, make sure you use the following cmake flags: `cmake -DCMAKE_BUILD_TYPE=Release -Dcppad_cxx_flags=-std=c++17`

## HPIPM
Using: https://github.com/Zolkin1/hpipm-cpp

# Running Code
Create the build folder. Then run
```
cmake --build . --config <Release/Debug/RelWithDebug>  --target <target> -- -j 30
```

# Profiling:
I personally like cachegrind more than perf.

## Valgrind
### Memcheck
Go to `cmake-build-relwithdebinfo` and run
```
valgrind --tool=memcheck ./mpc_app_test
```

### Cachegrind
Compile with optimizations and debug info.
Go to `cmake-build-relwithdebinfo` and run
```
valgrind --tool=cachegrind --cache-sim=yes ./mpc_app_test
```
To see the output:
```
cg_annotate cachegrind.out.<PID>
```

### Massif
*Currently appears to be encountering a segfault*.

In `cmake-build-relwithdebinfo` run
```
valgrind --tool=massif ./mpc_app_test
```

then use ms_print.

## Perf
See chat gpt
[//]: # (- Run with `-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer`)

[//]: # (- Get PID: `ps -eo pid,command | grep mpc_app_test | grep -v grep`)

[//]: # (- Recording: `sudo perf record -g -p PID`)

[//]: # (- Performance counter stats: `sudo perf stat -d -p PID`)

[//]: # (- Stop with Ctrl-C or when the program ends.)

[//]: # (- See data: `sudo perf report -i perf.data`)