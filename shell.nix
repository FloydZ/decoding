with import <nixpkgs> {};
let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {};
  pyEnv = mach-nix.mkPython rec {
    providers._default = "wheel,conda,nixpkgs,sdist";
    requirements = builtins.readFile ./requirements.txt;
  };
in
{ pkgs ? import <nixpkgs> {} }:
let 
  b = ''
    # NVIDIA Driver and CUDA setup
    export NVIDIA_VISIBLE_DEVICES=all
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export CUDA_VISIBLE_DEVICES=0


    # Library paths with specific NVIDIA driver
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    export LIBRARY_PATH=${pkgs.cudaPackages.cuda_cudart}/lib64:${pkgs.cudaPackages.cuda_cudart}/lib:$LIBRARY_PATH

    # OpenGL driver path
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH

    echo "CUDA C/C++ development environment ready"
    echo "GCC version in use:"
    gcc --version
    echo "NVCC version:"
    nvcc --version
  '';
in
stdenv.mkDerivation {
  name = "decoding";
  src = ./.;

  buildInputs = [
    pyenv
    gtest
    gbenchmark 
    git 
    cmake
    clang
    clang-tools
    llvmPackages.openmp
    llvm
    gcc
  ]++ (lib.optionals pkgs.stdenv.isLinux ([
    flamegraph
    gdb
    perf
    pprof
    valgrind

    # cuda
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cccl
    cudaPackages.cuda_gdb
    cudaPackages.cuda_cupti
    cudaPackages.nsight_compute
    cudaPackages.nsight_systems 
    cudaPackages.libcurand
    linuxPackages.nvidia_x11
  ]));
  shellHook = ''
    export NIX_ENFORCE_NO_NATIVE=0
    export NIX_HARDENING_ENABLE=0
  ''  + pkgs.lib.optionalString pkgs.stdenv.isLinux b; 
}

