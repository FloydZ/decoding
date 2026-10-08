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
}
