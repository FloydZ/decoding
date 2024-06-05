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

stdenv.mkDerivation {
  name = "decoding";
  src = ./.;

  buildInputs = [
    pyenv
  	gtest
	gbenchmark 
	git 
	cmake
	clang_17
    clang-tools_17
    llvmPackages_17.openmp
    llvm_17
	gcc
  ]++ (lib.optionals pkgs.stdenv.isLinux ([
   	flamegraph
   	gdb
    linuxKernel.packages.linux_6_6.perf
   	pprof
   	valgrind
   	massif-visualizer
    #TODO cudatoolkit
  ]));
}
