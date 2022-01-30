cmake_minimum_required(VERSION 3.20.1)
project(decoding)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

# needed for nvim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_CXX_STANDARD 20)
endif()

# This is like super stupid. But it needs to be done. Otherwise vim is not able to compile it.
#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CPP_COMPILER clang++)
include(ExternalProject)

set(CMAKE_VERBOSE_MAKEFILE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# load the external main project. This encloses all helper functions.
ExternalProject_Add(cryptanalysislib_project
        SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/cryptanalysislib
        CONFIGURE_COMMAND cd <SOURCE_DIR> && mkdir -p build
        COMMAND           cd <SOURCE_DIR> && cd build/ && cmake ..
        BUILD_COMMAND     cd <SOURCE_DIR> && cd build/ && make -j8
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )


# some helper variables
# possible values are: [-DVALUE_KARY, -DVALUE_BINARY]
# the first enables a k-ary implementation of the class 'Value'
# the second sets the class 'Value' on a binary vector
# set(VALUE_FLAG -DVALUE_BINARY)
# set(VALUE_FLAG -DVALUE_KARY)
set(VALUE_FLAG "")



#set(MALLOC_LIBRARY tcmalloc)
set(MALLOC_LIBRARY "")

# possible sanitzers = -fsanitize=[address,leak,thread,memory,undefined]
#set(SANITIZE_FLAG "-fsanitize=address")
set(SANITIZE_FLAG "")

#set(ADDITIONAL_FLAGS "-DUSE_AVX2  -fno-inline")
set(ADDITIONAL_FLAGS "-DUSE_AVX2")

set(ALLOWED_WARNINGS " -Wno-unused-function -Wno-unused-local-typedef -Wno-overlength-strings -Wno-unknown-pragmas -Wno-unused-variable")

set(DEBUG_FLAGS "-DDEBUG -Wall -Wpedantic -pedantic -march=native ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAG} ${ADDITIONAL_FLAGS}")
set(OPTIMIZE_FLAGS "-Ofast -ffast-math -ftree-vectorize -funroll-loops -Wall -Wpedantic -pedantic -march=native -flto -fwrapv -fomit-frame-pointer -fno-exceptions ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAG} ${ADDITIONAL_FLAGS}")
#set(OPTIMIZE_FLAGS "-march=native -O3 -flto -DNDEBUG ${VALUE_FLAG}")
set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
set(GCC_HELPER_FLAGS -fvar-tracking-assignments)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(DEBUG_FLAGS "-std=c++20 ${DEBUG_FLAGS}")
        set(OPTIMIZE_FLAGS "-std=c++20 ${OPTIMIZE_FLAGS}")
endif()

# this flag is like super important for performance benchmarking, because otherwise we don't see the exact functions
# called in the flamegraphs
set(COMPILE_TEST_FLAGS -g)

# uncomment the following line to enable the BOLT optimiser.
#set(LINK_TEST_FLAGS -lgtest -pthread gmp gmpxx m4ri m png -fopenmp -Xlinker --emit-relocs )
set(LINK_TEST_FLAGS -lgtest -pthread gmp gmpxx m4ri m png -lomp)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(LINK_TEST_FLAGS -lgtest -lgtest_main -pthread gmp gmpxx m png -lomp m4ri.a )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # if using GCC
        set(DEBUG_FLAGS "${DEBUG_FLAGS} ${GCC_HELPER_FLAGS} -O0")
endif()

# Include tcmalloc if you want
if(MALLOC_LIBRARY STREQUAL "tcmalloc")
        # sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
        # for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
        find_package(TCMalloc)
        if(NOT TCMALLOC_LIBRARIES)
                MESSAGE(FATAL_ERROR "TCMALLOC_LIBRARIES was set but TCMALLOC was not found!")
        endif()

        set(OPTIMIZE_FLAGS "-march=native -O3 -flto -DNDEBUG ${VALUE_FLAG} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
        set(LINK_TEST_FLAGS -lgtest -pthread gmp gmpxx m4ri m png ${TCMALLOC_LIBRARIES})
else()
        SET(TCMALLOC_LIBRARIES "")
endif()

#set openmp
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        if ( CMAKE_CXX_COMPILER_ID STREQUAL "clang")
                set(DEBUG_FLAGS "${DEBUG_FALGS} -Xclang -fopenmp")
                set(OPTIMIZE_FLAGS "${OPTIMIZE_FLAGS}  -Xclang -fopenmp")
        endif()
else()
        set(DEBUG_FLAGS "${DEBUG_FLAGS} -fopenmp")
        set(OPTIMIZE_FLAGS "${OPTIMIZE_FLAGS} -fopenmp")
endif()

# Set additionally our own debug/optimisation compiler/linkage flags. Maybe not he nicest way, but by far the easiest.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")


#TODO include cuda if possible
set(USE_CUDA "1")
if(USE_CUDA)
        enable_language(CUDA)

        #ExternalProject_Add(nvbench_project
        #        SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/nvbench
        #        CONFIGURE_COMMAND cd <SOURCE_DIR> && git checkout ff50759 && mkdir -p build
        #        COMMAND           cd <SOURCE_DIR> && cd build/ && cmake -DCMAKE_CUDA_ARCHITECTURES=60 -DCUDA_ARCHITECTURES=60-real -DNVBench_ENABLE_CUPTI=ON ..
        #        BUILD_COMMAND     cd <SOURCE_DIR> && cd build/ && make -j8
        #        INSTALL_COMMAND   ""
        #        TEST_COMMAND      ""
        #        )

        # global include path for dependencies
        # maybe a little ugly, but the easiest way to include `cuda_runtime.h`
        # as far as i know this is only needed for arch linux.
        include_directories(/opt/cuda/samples/common/inc)
        include_directories(/opt/cuda/include)
        include_directories(deps/cudacommon)
        include_directories(deps/cuCollections/include)
        # cannot be included in general, because this some how added automatically a second main function
        # include_directories(deps/nvbench)

        #for ubuntu
        include_directories(/usr/local/cuda/samples/common/inc)

        # set the cuda standard
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 17)
                set(CMAKE_CUDA_STANDARD_REQUIRED True)
        endif()


        # fall back, if one does not directly have a CUDA rdy card, but rather a optimus (internal laptop) card.
        #if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 61)
        #endif()

        # this is needed to get the libary created
        #add_subdirectory(deps/nvbench)
        #add_executable(test_cuda tests/test_cuda.cu)
        #target_link_libraries(test_cuda ${LINK_TEST_FLAGS} fplll m4ri gmpxx gmp)
endif()

# include every subpackage which is included in `cryptanalysislib`
include_directories("deps/cryptanalysislib/deps/fplll/")
include_directories("deps/cryptanalysislib/deps/b63/src")
include_directories("deps/cryptanalysislib/deps/m4ri/m4ri")
include_directories("deps/cryptanalysislib/deps/m4ri")
include_directories("deps/cryptanalysislib/deps/ska_sort")
include_directories("deps/cryptanalysislib/src")

include_directories("src")
include_directories("test/decoding/challenges")
include_directories("test/mceliece/challenges")
include_directories("/usr/local/include")

link_directories("deps/cryptanalysislib/build")
link_directories("deps/cryptanalysislib/deps/m4ri/.libs")
link_directories("/usr/local/lib")

add_subdirectory(test)
add_subdirectory(bench)

# add the main target running our main algorithm
add_executable(main main.cpp)
target_compile_options(main PUBLIC -g)        # for benchmark symbols
target_link_libraries(main ${LINK_TEST_FLAGS})
#add_dependencies(main cryptanalysislib_project)

add_custom_target (
        LLVM_PROFDATA
        COMMAND llvm-profdata-13 merge -output=default.profdata default_*.profraw
)

add_executable(main_profile1 main.cpp)
target_compile_options(main_profile1 PUBLIC -fprofile-generate -DUSE_LOOPS=2)
target_link_libraries(main_profile1 ${LINK_TEST_FLAGS} -fprofile-instr-generate)

add_executable(main_profile2 main.cpp)
target_compile_options(main_profile2 PUBLIC -fprofile-use -fprofile-correction -DCHALLENGE)
target_link_libraries(main_profile2 ${LINK_TEST_FLAGS})
add_dependencies(main_profile2 LLVM_PROFDATA)

#if (CMAKE_CXX_COMPILER_ID STREQUAL "clang")
        add_executable(main_PGO_instrumentation_first main.cpp)
        target_compile_options(main_PGO_instrumentation_first PUBLIC -fprofile-instr-generate -DUSE_LOOPS=2)
        target_link_libraries(main_PGO_instrumentation_first ${LINK_TEST_FLAGS} -fprofile-instr-generate)
        #target_link_options(main_PGO_instrumentation_first PUBLIC -fprofile-instr-generate)

        add_executable(main_PGO_instrumentation_second main.cpp)
        target_compile_options(main_PGO_instrumentation_second PUBLIC -fprofile-instr-use=code.profdata -DCHALLENGE)
        target_link_libraries(main_PGO_instrumentation_second ${LINK_TEST_FLAGS} -fprofile-instr-use=${CMAKE_SOURCE_DIR}/cmake-build-release/code.profdata)
        #target_link_options(main_PGO_instrumentation_second PUBLIC -fprofile-instr-use=${CMAKE_SOURCE_DIR}/cmake-build-release/code.profdata)
#endif()

#add_executable(main_polly main.cpp)
#target_compile_options(main_polly PUBLIC -mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM)
#target_link_libraries(main_polly ${LINK_TEST_FLAGS})

add_executable(main_bolt main.cpp)
#target_compile_options(main_bolt PUBLIC --emit-relocs)
target_link_libraries(main_bolt ${LINK_TEST_FLAGS})
target_link_options(main_bolt PUBLIC -Xlinker --emit-relocs)
