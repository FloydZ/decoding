cmake_minimum_required(VERSION 3.22)
project(decoding)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

# needed for nvim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

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

add_custom_target (
        CRYPTANALYSISLIB_BUILD_SCRIPT
        COMMAND ${CMAKE_SOURCE_DIR}/setup.sh
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
set(SANITIZE_FLAGS "")


set(ALLOWED_WARNINGS " -Wno-unused-function -Wno-unused-local-typedef -Wno-overlength-strings -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-local-typedefs")


if(APPLE)
    include_directories(/opt/homebrew/opt/libomp/include)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/opt/libomp/lib)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/opt/homebrew/lib)

    set(ADDITIONAL_FLAGS "-DUSE_ARM")
    set(DEBUG_FLAGS "-g -Og -DDEBUG -fopenmp ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-O3 -flto -DNDEBUG -fno-exceptions -fopenmp ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAG} ${ADDITIONAL_FLAGS}")
else()
    #set(ADDITIONAL_FLAGS "-DUSE_AVX2  -fno-inline")
    #set(ADDITIONAL_FLAGS "-DUSE_AVX2 -Werror=dangling-reference")
    set(ADDITIONAL_FLAGS "-DUSE_AVX2")

    set(AVX512_FLAGS "-DUSE_AVX512 -mavx512f -mavx512dq  -mavx512bf16 -mavx512cd  -mavx512bw -mavx512vl -mavx512vbmi -mavx512ifma -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq")
    set(AVX2_FLAGS "-mavx2 -mavx -mbmi -mbmi2")
    set(DEBUG_FLAGS "-g -O0 -DDEBUG -Wall -Wpedantic -pedantic -march=native -fopenmp ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-g -march=native -mavx2 -mavx -mbmi2 -O3 -flto -DNDEBUG -fopenmp -ffast-math -ftree-vectorize -funroll-loops -fno-exceptions ${AVX2_FLAGS} ${AVX512_FLAGS} ${ALLOWED_WARNINGS} ${VALUE_FLAG} ${SANITIZE_FLAGS} ${ADDITIONAL_FLAGS}")
endif()

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
        set(LINK_TEST_FLAGS -lgtest -lgtest_main -pthread gmp gmpxx m png m4ri.a )
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

# set openmp
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        if ( CMAKE_CXX_COMPILER_ID STREQUAL "clang")
                set(DEBUG_FLAGS "${DEBUG_FLAGS} -Xclang -fopenmp")
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


SET(CMAKE_CXX_FLAGS_RELEASE_PROFILE -fprofile-instr-generate=code.profraw -DUSE_LOOPS=1000)
set(CMAKE_CXX_FLAGS_RELEASE_ANALYSIS -fdiagnostics-show-hotness -fdiagnostics-fixit-info  -fprofile-instr-use=code.profdata  -fsave-optimization-record -foptimization-record-file=analysis.txt -Rpass-analysis -DUSE_LOOPS=1000)

macro(use_host_compiler)
    if (${CURRENT_COMPILER} STREQUAL "NATIVE")
        # Save current native flags
        set(NATIVE_C_FLAGS ${CMAKE_C_FLAGS} CACHE STRING "GCC flags for the native compiler." FORCE)

        # Change compiler
        set(CMAKE_SYSTEM_NAME ${CMAKE_HOST_SYSTEM_NAME})
        set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
        set(CMAKE_C_COMPILER ${HOST_C_COMPILER})
        set(CMAKE_C_FLAGS ${HOST_C_FLAGS})
        set(CURRENT_COMPILER "HOST" CACHE STRING "Which compiler we are using." FORCE)
    endif()
endmacro()


macro(use_sclang_compiler)
    #if (${CURRENT_COMPILER} STREQUAL "SLANG")
        # Save current host flags
        set(SCLANG_C_FLAGS ${CMAKE_C_FLAGS} CACHE STRING "GCC flags for the host compiler." FORCE)

        # Change compiler
        set(CMAKE_SYSTEM_NAME "SCLANG")
        #set(CMAKE_SYSTEM_PROCESSOR "/home/duda/Downloads/souper/build'slang")
        set(CMAKE_CXX_COMPILER "/home/duda/Downloads/souper/build/sclang++")
        #set(CMAKE_CXX_FLAGS ${NATIVE_C_FLAGS})
        set(CURRENT_COMPILER "NATIVE" CACHE STRING "Which compiler we are using." FORCE)
    #endif()
endmacro()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        ExternalProject_Add(nvbench_project
                SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/nvbench
                CONFIGURE_COMMAND cd <SOURCE_DIR> && git checkout ff50759 && mkdir -p build
                COMMAND           cd <SOURCE_DIR> && cd build/ && cmake -DCMAKE_CUDA_ARCHITECTURES=60 -DCUDA_ARCHITECTURES=60-real -DNVBench_ENABLE_CUPTI=ON ..
                BUILD_COMMAND     cd <SOURCE_DIR> && cd build/ && make -j8
                INSTALL_COMMAND   ""
                TEST_COMMAND      ""
                )

        # global include path for dependencies
        # maybe a little ugly, but the easiest way to include `cuda_runtime.h`
        # as far as i know this is only needed for arch linux.
        include_directories(/opt/cuda/samples/common/inc)
        # again only for arch... fuck NVIDIA
        include_directories(/opt/cuda/samples/cuda-samples/Common)
        include_directories(/usr/local/cuda/samples/cuda-samples/Common)
        include_directories(/home/floyd.zweydinger/cuda-samples/Common)
        include_directories(/opt/cuda/include)
        include_directories(deps/cudacommon)
        include_directories(deps/cuCollections/include)
        include_directories(deps/cuda-samples/Common)
        # cannot be included in general, because this some how added automatically a second main function
        # include_directories(deps/nvbench)

        #for ubuntu
        include_directories(/usr/local/cuda/samples/common/inc)

        # set the cuda standard
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 17)
                set(CMAKE_CUDA_STANDARD_REQUIRED True)
        endif()
		
		set(CUDA_OPTIMIZE_FLAGS "-march=native -O3 --resource-usage --resource-usage -Xptxas -O3 -use_fast_math -extra-device-vectorization -restrict -Xptxas -warn-spills")
        #set(CUDA_OPTIMIZE_FLAGS "-march=native -O3 --resource-usage -ccbin=/usr/bin/gcc-9 --resource-usage -Xptxas -O3 -use_fast_math -extra-device-vectorization -restrict -Xptxas -warn-spills")

        # fall back, if one does not directly have a CUDA rdy card, but rather a optimus (internal laptop) card.
        #if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 75 61)
        #endif()

        # this is needed to get the libary created
        add_subdirectory(deps/nvbench)
        #add_executable(test_cuda tests/test_cuda.cu)
        #target_link_libraries(test_cuda ${LINK_TEST_FLAGS} fplll m4ri gmpxx gmp)


		add_executable(main_cuda main.cu)
		target_link_libraries(main_cuda m4ri)
		# target_compile_options(main_cuda PUBLIC -march=native -O3 -ccbin=/usr/bin/gcc-9 --resource-usage -use_fast_math -extra-device-vectorization -restrict -Xptxas -O3 ${COMPILE_TEST_FLAGS})
		# target_compile_options(main_cuda PUBLIC -march=native -O3 --resource-usage -use_fast_math -extra-device-vectorization -restrict -Xptxas -O3 ${COMPILE_TEST_FLAGS})
        target_compile_options(main_cuda PUBLIC -march=native -O3 ${COMPILE_TEST_FLAGS})
        #target_compile_options(main_cuda PUBLIC -G -g -DDEBUG  ${COMPILE_TEST_FLAGS})
endif()

# include every subpackage which is included in `cryptanalysislib`
include_directories("deps/cryptanalysislib/deps/fplll/")
include_directories("deps/cryptanalysislib/deps/b63/src")
include_directories("deps/cryptanalysislib/deps/m4ri/m4ri")
include_directories("deps/cryptanalysislib/deps/m4ri")
include_directories("deps/cryptanalysislib/deps/ska_sort")
include_directories("deps/cryptanalysislib/deps/Static-Sort/include")
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

add_custom_target(
    TOUCH
	COMMAND touch ${CMAKE_BINARY_DIR}/code.profraw ${CMAKE_BINARY_DIR}/code.profdata
)

add_custom_target (
    LLVM_PROFDATA
    COMMAND llvm-profdata merge -output=code.profdata code.profraw
	DEPENDS TOUCH
)


# add the main target running our main algorithm
add_executable(main main.cpp)
#target_compile_options(main PUBLIC ${OPTIMIZE_FLAGS})        # for benchmark symbols
#target_compile_options(main PUBLIC -g)        # for benchmark symbols
target_link_libraries(main ${LINK_TEST_FLAGS})
#add_dependencies(main cryptanalysislib_project)

add_executable(main_analysis1 main.cpp)
target_link_libraries(main_analysis1 ${LINK_TEST_FLAGS} -fprofile-instr-generate=code.profraw)
target_compile_options(main_analysis1 PRIVATE ${CMAKE_CXX_FLAGS_RELEASE_PROFILE})

add_executable(main_analysis2 main.cpp)
target_link_libraries(main_analysis2 ${LINK_TEST_FLAGS})
target_compile_options(main_analysis2 PRIVATE ${CMAKE_CXX_FLAGS_RELEASE_ANALYSIS})
add_dependencies(main_analysis2 LLVM_PROFDATA)

add_executable(main_profile1 main.cpp)
target_compile_options(main_profile1 PUBLIC -fprofile-generate -DUSE_LOOPS=100)
target_link_libraries(main_profile1 ${LINK_TEST_FLAGS} -fprofile-instr-generate)

add_executable(main_profile2 main.cpp)
target_compile_options(main_profile2 PUBLIC -fprofile-use -fprofile-correction -DCHALLENGE)
target_link_libraries(main_profile2 ${LINK_TEST_FLAGS})
add_dependencies(main_profile2 LLVM_PROFDATA)

if (CMAKE_CXX_COMPILER_ID STREQUAL "clang")
        add_executable(main_PGO_instrumentation_first main.cpp)
        target_compile_options(main_PGO_instrumentation_first PUBLIC -fprofile-instr-generate -DUSE_LOOPS=2)
        target_link_libraries(main_PGO_instrumentation_first ${LINK_TEST_FLAGS} -fprofile-instr-generate)
        #target_link_options(main_PGO_instrumentation_first PUBLIC -fprofile-instr-generate)

        add_executable(main_PGO_instrumentation_second main.cpp)
        target_compile_options(main_PGO_instrumentation_second PUBLIC -fprofile-instr-use=code.profdata -DCHALLENGE)
        target_link_libraries(main_PGO_instrumentation_second ${LINK_TEST_FLAGS} -fprofile-instr-use=${CMAKE_SOURCE_DIR}/cmake-build-release/code.profdata)
        #target_link_options(main_PGO_instrumentation_second PUBLIC -fprofile-instr-use=${CMAKE_SOURCE_DIR}/cmake-build-release/code.profdata)
endif()



set(USE_POLLY "0")
if(USE_POLLY)
	#SRC: https://polly.llvm.org/docs/UsingPollyWithClang.html
    #cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=/home/duda/Downloads/BOLT/build/bin/clang++ ..
    #cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=/home/duda/Downloads/BOLT/build/bin/clang++  -DCMAKE_LINKER=clang++-14 -DCMAKE_CXX_LINK_EXECUTABLE='/usr/bin/clang++-14 <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>'  ..
	add_executable(main_polly main.cpp)
	target_compile_options(main_polly PUBLIC
		"SHELL:-mllvm -polly"
		"SHELL:-mllvm -polly-parallel"
		"SHELL:-mllvm -polly-omp-backend=LLVM"      #GNU
		"SHELL:-mllvm -polly-scheduling=runtime"    #dynamic, static, guided
		"SHELL:-mllvm -polly-vectorizer=stripmine"
        ${OPTIMIZE_FLAGS}
	)

    target_link_libraries(main_polly ${LINK_TEST_FLAGS})

    #set_target_properties(main_polly PROPERTIES CMAKE_LINKER ${CMAKE_CXX_COMPILER})
    set_target_properties(main_polly PROPERTIES CMAKE_LINKER /usr/bin/clang)
    set_target_properties(main_polly PROPERTIES CMAKE_C_LINK_EXECUTABLE "/usr/bin/clang <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    #set(CMAKE_C_LINK_EXECUTABLE "clang <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    set_target_properties(main_polly PROPERTIES CMAKE_CXX_LINK_EXECUTABLE "/usr/bin/clang++ <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")


add_executable(main_cuda main.cpp)
endif()

set(USE_BOLT "0")
if(USE_BOLT)
    # cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=/home/duda/Downloads/BOLT/build/bin/clang++  ..
    SET(LLVM_BOLT_PATH "/home/duda/Downloads/BOLT/build/bin/llvm-bolt")
	#add_custom_target (
	#		LLVM_BOLT
	#		COMMAND ${LLVM_BOLT_PATH} main_bolt1 -instrument -jump-tables=move -o main_bolt_out -align-macro-fusion=all
	#)

	add_executable(main_bolt1 main.cpp)
	target_compile_options(main_bolt1 PUBLIC -DUSE_LOOPS=10000)
	target_link_libraries(main_bolt1 ${LINK_TEST_FLAGS} )
	target_link_options(main_bolt1 PUBLIC -Xlinker --emit-relocs)

    add_custom_command(TARGET main_bolt1
        POST_BUILD
        COMMAND ${LLVM_BOLT_PATH} main_bolt1 -instrument -jump-tables=move -o main_bolt_instrument -align-macro-fusion=all
    )

	add_executable(main_bolt2 main.cpp)
	target_compile_options(main_bolt2 PUBLIC  -DCHALLENGE)
	target_link_libraries(main_bolt2 ${LINK_TEST_FLAGS})
    target_link_options(main_bolt2 PUBLIC -Xlinker --emit-relocs)

    add_custom_command(TARGET main_bolt2
        POST_BUILD
        #COMMAND ${LLVM_BOLT_PATH} main_bolt2 -o main_bolt.bolt -data=/tmp/prof.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats -jump-tables=move -align-macro-fusion=all --simplify-conditional-tail-calls --peepholes=all --hugify --icp-eliminate-loads --icf
        COMMAND ${LLVM_BOLT_PATH} main_bolt2 -o main_bolt.bolt -data=/tmp/prof.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats -jump-tables=move -align-macro-fusion=all
    )


    set(USE_POLLY "1")
    if(USE_POLLY)
        add_executable(main_polly_bolt1 main.cpp)
        target_compile_options(main_polly_bolt1 PUBLIC
                "SHELL:-mllvm -polly"
                "SHELL:-mllvm -polly-parallel"
                "SHELL:-mllvm -polly-omp-backend=LLVM"      #GNU
                "SHELL:-mllvm -polly-scheduling=runtime"    #dynamic, static, guided
                "SHELL:-mllvm -polly-vectorizer=stripmine"
                ${OPTIMIZE_FLAGS}
                )

        target_compile_options(main_polly_bolt1 PUBLIC -DUSE_LOOPS=10000)
        target_link_libraries(main_polly_bolt1 ${LINK_TEST_FLAGS} )
        target_link_options(main_polly_bolt1 PUBLIC -Xlinker --emit-relocs)

        set_target_properties(main_polly_bolt1 PROPERTIES CMAKE_LINKER /usr/bin/clang)
        set_target_properties(main_polly_bolt1 PROPERTIES CMAKE_C_LINK_EXECUTABLE "/usr/bin/clang <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
        set_target_properties(main_polly_bolt1 PROPERTIES CMAKE_CXX_LINK_EXECUTABLE "/usr/bin/clang++ <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

        add_custom_command(TARGET main_polly_bolt1
            POST_BUILD
            COMMAND ${LLVM_BOLT_PATH} main_bolt1 -instrument -jump-tables=move -o main_bolt_instrument -align-macro-fusion=all
        )

        add_executable(main_polly_bolt2 main.cpp)
        target_compile_options(main_polly_bolt2 PUBLIC -DCHALLENGE)
        target_link_libraries(main_polly_bolt2 ${LINK_TEST_FLAGS})
        target_link_options(main_polly_bolt2 PUBLIC -Xlinker --emit-relocs)

        add_custom_command(TARGET main_polly_bolt2
                POST_BUILD
                COMMAND ${LLVM_BOLT_PATH} main_bolt2 -o main_bolt.bolt -data=/tmp/prof.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats -jump-tables=move -align-macro-fusion=all
                )

    endif()
endif()

