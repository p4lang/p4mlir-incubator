# P4MLIR

P4MLIR is an (experimental) effort looking to leverage [MLIR](https://mlir.llvm.org/) in building [P4](https://p4.org/) compilers.

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

#### Build P4MLIR from source

```shell
./build_tools/build_p4mlir.sh
```

### Testing

```shell
cd build
ninja check-p4mlir
```
