git submodule sync
git submodule update --init
cd deps/cryptanalysislib
git checkout 4d3fd5da85b301d08bee25f21eb9fc5d00312253
git submodule sync
git submodule update --init
cd deps/m4ri
git checkout 042cab8
autoreconf --install
./configure
make -j8
cp ../patches/patch.diff ./patch.diff
git apply patch.diff
cd ../../
cd deps/fplll
git checkout 5.3.0
./autogen.sh
./configure
make -j8
cd ../../
cd ../../
mkdir -p cmake-build-release && cd cmake-build-release && cmake -DCMAKE_BUILD_TYPE=Release .. && cd ..
