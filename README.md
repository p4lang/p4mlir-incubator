# P4MLIR

**WARNING**: This is an experimental project in incubation, not ready for any serious use yet.

P4MLIR aims to explore leveraging [MLIR](https://mlir.llvm.org/) in building [P4](https://p4.org/) compilers.

P4MLIR is structured as both a [standalone MLIR project](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone), and a [P4C extension](https://github.com/fruffy/p4dummy). Once it's mature enough, the plan is to merge it into the [P4C repo](https://github.com/p4lang/p4c).

## Developer guide

### Build from source

#### Check out the code

```shell
git clone --recursive git@github.com:p4lang/p4mlir.git
cd p4mlir
```

If you forgot `--recursive`, additionally run:

```shell
git submodule update --init --recursive
```

#### Build MLIR from source

Install common LLVM/MLIR requirements following the [instructions](https://mlir.llvm.org/getting_started/). Or use our script if you are on Ubuntu:

```shell
./build_tools/ubuntu_install_mlir_requirements.sh
```

Then build MLIR from source and install it:

```shell
./build_tools/build_mlir.sh
```

Sometimes this script will stop in the middle and report some failures. Re-running the script will continue the build process and usually go past the last failure. Eventually the build shall finish successfully.

Some MLIR regression tests have already been performed during the build process. If you want to run them again, do the following:

```shell
cd third_party/llvm-project/build
ninja check-mlir
```

*IMPORTANT*: P4C uses C++ RTTI and Exceptions. These are turned off by default
in LLVM and therefore LLVM/MLIR prebuilt binaries that is likely available from
your distribution will be not usable with P4MLIR. One needs to build LLVM/MLIR
from source enabling RTTI (`-DLLVM_ENABLE_RTTI=ON`) and exceptions
(`-DLLVM_ENABLE_EH=ON`).

#### Build P4C with P4MLIR extension from source

```shell
./build_tools/build_p4c_with_p4mlir_ext.sh
```

### Testing

```shell
cd third_party/p4c/build
ninja check-p4mlir
```
