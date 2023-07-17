with import <nixpkgs> {};
let
  my-python = pkgs.python3;
  python-with-my-packages = my-python.withPackages (p: with p; [
	scipy
    python-lsp-server
  ]);
in
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "decoding";
  src = ./.;

  buildInputs = [ 
  	python-with-my-packages
  	gtest
	gbenchmark 
	git 
	libtool 
	autoconf 
	automake 
	autogen 
	gnumake 
	cmake 
	clang_14
    clang-tools_14
    llvmPackages_14.openmp
	ripgrep
	gmp
	libpng
	mpfr
	fplll
	tbb
	#sage
  ]++ (lib.optionals pkgs.stdenv.isLinux ([
   	flamegraph
   	gdb
    linuxKernel.packages.linux_6_3.perf
   	pprof
   	valgrind
   	massif-visualizer
  ]));
}
