cmake_minimum_required(VERSION 3.20)
project(decoding)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(USE_TCMALLOC "Enable tcmalloc" OFF)
option(USE_SANITIZER "Enable memory sanitizer" ON)
option(USE_NOINLINE "Enable -fno-inline" OFF)
option(USE_LTO "Enable -flto" OFF)
option(USE_PROFILE "Enable profile guided optmization" OFF)
option(USE_POLLY "Enable polly loop optimization" OFF)
option(USE_BOLT "Enable bolt binary optmization" OFF)
option(USE_TRACY "Enable tracy profiling" ON)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# needed for nvim autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# import needed  
include(FetchContent)
include(ExternalProject)
include(CheckCXXCompilerFlag) 	# check for compiler optimization flags
include(CheckIPOSupported)  	# check for -flto
include(CheckLanguage) 			# needed for `check_language(CUDA)`

# load the external main project. This encloses all helper functions.
ExternalProject_Add(cryptanalysislib_project
        SOURCE_DIR        ${CMAKE_SOURCE_DIR}/deps/cryptanalysislib
        CONFIGURE_COMMAND cd <SOURCE_DIR> && mkdir -p build
        COMMAND           cd <SOURCE_DIR> && cd build/ && cmake ..
        BUILD_COMMAND     cd <SOURCE_DIR> && cd build/ && make -j8
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)


include("deps/cryptanalysislib/deps/cmake_optimize_for_architecture/HostOptimizations.cmake")

check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag(-fno-inline COMPILER_SUPPORTS_NO_INLINE)
check_cxx_compiler_flag(-fsanitize=address COMPILER_SUPPORTS_SANITIZE_ADDRESS) # TODO support not recognized
check_cxx_compiler_flag(-fsanitize=memory COMPILER_SUPPORTS_SANITIZE_MEMORY)
check_cxx_compiler_flag(-fconstexpr-steps=9999999 COMPILER_SUPPORTS_CONSTEXPR_COST)
check_cxx_compiler_flag(-fno-exceptions COMPILER_SUPPORTS_NO_EXCEPTION)
check_ipo_supported(RESULT lto_supported OUTPUT error)
check_cxx_compiler_flag(-latomic LINKER_SUPPORTS_ATOMIC) # TODO

# find needed packages: the good thing only is: we only have a single external
# dependencies. 
enable_testing()
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest) 
find_package(benchmark)
find_package(Git QUIET)

# init the submodules
FetchContent_Declare(
	tracy
	GIT_REPOSITORY 	https://github.com/wolfpld/tracy
	GIT_TAG 		master
	GIT_SHALLOW 	TRUE
	GIT_PROGRESS 	TRUE
)
FetchContent_MakeAvailable(tracy)

# global include directories
include_directories("deps/b63/include/b63")
include_directories("src")
include_directories("/usr/local/include")

# global link directories
link_directories("/usr/local/lib")

# two possible options. 'tcmalloc' is the standard google malloc implementation
if(USE_TCMALLOC)
	message(STATUS "Using tcmalloc")
	set(MALLOC_LIBRARY_FLAGS "tcmalloc")
endif()

#if(USE_SANITIZER AND COMPILER_SUPPORTS_SANITIZE_ADDRESS)
	message(STATUS "Using address sanitize")
	# possible sanitizers = -fsanitize=[address,leak,thread,memory,undefined]
	set(SANITIZER_FLAGS "-fsanitize=address -fsanitize=pointer-compare -fno-omit-frame-pointer")
#endif()

if(USE_NOINLINE AND COMPILER_SUPPORTS_NO_INLINE)
	message(STATUS "not inlining")
	set(INLINE_FLAGS "-fno-inline")
endif()

if (COMPILER_SUPPORTS_CONSTEXPR_COST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=99999999")
endif()

if(lto_supported AND USE_LTO)
  message(STATUS "IPO / LTO enabled")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(ALLOWED_WARNINGS " -Wno-unknown-pragmas -Wno-c99-designator -Wno-invalid-constexpr -Wno-unused-function -Wno-zero-length-array")

message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") 
	message(STATUS "ARM CPU detected")
    set(ADDITIONAL_FLAGS "-DUSE_ARM -flax-vector-conversions -march=native")
	set(DEBUG_FLAGS "-g -Wall -Wextra -fpermissive -DDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
    set(OPTIMIZE_FLAGS "-O3 -DNDEBUG  ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${ADDITIONAL_FLAGS}")
else()

    set(DEBUG_FLAGS "-g -O0 -Wall -Wextra -DDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${SANITIZER_FLAGS} ${ADDITIONAL_FLAGS}")
	set(OPTIMIZE_FLAGS "-g -march=native -DNDEBUG ${INLINE_FLAGS} ${ALLOWED_WARNINGS} ${ADDITIONAL_FLAGS}")
endif()

set(LINK_TEST_FLAGS 
	GTest::gtest 
	GTest::gtest_main 
	Threads::Threads 
	OpenMP::OpenMP_CXX
)

if(USE_TRACY)
	message(STATUS "Using tracy")
	list(APPEND LINK_TEST_FLAGS TracyClient)
	# list(APPEND LINK_BENCH_FLAGS TracyClient)
endif()

# Set additionally our own debug/optimisation compiler/linkage flags. Maybe not he nicest way, but by far the easiest.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
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

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # global include path for dependencies
    # maybe a little ugly, but the easiest way to include `cuda_runtime.h`
    # as far as i know this is only needed for arch linux.
    include_directories(/opt/cuda/samples/common/inc)
    # again only for arch... fuck NVIDIA
    include_directories(/opt/cuda/samples/cuda-samples/Common)
    include_directories(/usr/local/cuda/samples/cuda-samples/Common)
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
    #add_executable(test_cuda tests/test_cuda.cu)
    #target_link_libraries(test_cuda ${LINK_TEST_FLAGS} fplll m4ri gmpxx gmp)


	#add_executable(main_cuda main.cu)
	#target_link_libraries(main_cuda m4ri)
    #target_compile_options(main_cuda PUBLIC -march=native -O3 ${COMPILE_TEST_FLAGS})
endif()

# include every subpackage which is included in `cryptanalysislib`
include_directories("deps/cryptanalysislib/deps/b63/include/b63")
include_directories("deps/cryptanalysislib/src")

include_directories("src")
include_directories("test/decoding/challenges")
include_directories("test/mceliece/challenges")
include_directories("/usr/local/include")

link_directories("/usr/local/lib")

#TODO 
# if (benchmark_FOUND)
# 	add_subdirectory(bench)
# endif()
if (GTest_FOUND)
	add_subdirectory(tests)
endif()

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
#TODO add_executable(main main.cpp)
# target_link_libraries(main ${LINK_TEST_FLAGS})

if(USE_PROFILE)
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
endif()

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



if(USE_POLLY)
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
endif()

