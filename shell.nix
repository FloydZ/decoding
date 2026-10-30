with import <nixpkgs> {};
let 
  unstable_pkgs = import <nixos-unstable> {};
  ## Path setup
  #export PATH="${pkgs.gcc}/bin:$PATH"
  #export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH

  ## CUDA setup
  #export CUDAHOSTCXX="${pkgs.gcc}/bin/g++"
  #export CUDA_HOST_COMPILER="${pkgs.gcc}/bin/gcc"
  #export CUDA_HOME=${pkgs.cudaPackages.cuda_cudart}
  #export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}
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
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "rust_lll";
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
  ]));

  shellHook = ''
    export NIX_ENFORCE_NO_NATIVE=0
    export NIX_HARDENING_ENABLE=0

    SOURCE_DATE_EPOCH=$(date +%s)
    VENV=.venv
    if test ! -d $VENV; then
      virtualenv $VENV
    fi
    source ./$VENV/bin/activate
  '' + pkgs.lib.optionalString pkgs.stdenv.isLinux b;
}

