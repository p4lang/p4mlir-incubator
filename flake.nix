# SPDX-FileCopyrightText: 2026 The P4 Language Consortium
#
# SPDX-License-Identifier: Apache-2.0

{
  description = "P4MLIR development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt);

      devShells = forAllSystems (
        pkgs:
        let
          inherit (pkgs) llvmPackages;

          # Compile with clang, as build_tools/*.sh do by default. On Linux the
          # stock clang wrapper links with GNU ld; pair it with LLVM's bintools
          # instead so that `ld` is lld, like the Ubuntu setup.
          clang =
            if pkgs.stdenv.isLinux then
              llvmPackages.clang.override { inherit (llvmPackages) bintools; }
            else
              llvmPackages.clang;

          mkShell = pkgs.mkShell.override { stdenv = pkgs.overrideCC pkgs.stdenv clang; };
        in
        {
          default = mkShell {
            # Build tools, mirroring build_tools/ubuntu_install_mlir_requirements.sh.
            nativeBuildInputs = with pkgs; [
              bison
              ccache
              cmake
              flex
              git
              ninja
              pkg-config
              python3
              # Developer tools for the linter and formatter configs in the repo.
              llvmPackages.clang-tools # clang-format, clang-tidy, clangd
              cpplint
              reuse
            ];

            # Libraries. Listing them here (rather than as tools) is what puts
            # them on CMake's search path and the compiler's include path.
            buildInputs = with pkgs; [
              # LLVM & MLIR dependencies
              zlib
              # P4C dependencies
              boost
              flex # libfl
            ];

            # Keep the compile flags identical to a non-Nix build. The nixpkgs
            # compiler wrapper would otherwise inject hardening flags (fortify,
            # stack protector, ...) that the Ubuntu toolchain does not use.
            hardeningDisable = [ "all" ];

            # build_mlir.sh defaults to -DLLVM_ENABLE_LLD=ON, which makes clang
            # execute a bare `ld.lld` and bypass the nixpkgs linker wrapper,
            # losing the library search paths and rpaths it injects. `ld` is
            # the wrapped default linker: lld on Linux, ld64 on macOS.
            LLVM_USE_LINKER = "ld";
          };
        }
      );
    };
}
