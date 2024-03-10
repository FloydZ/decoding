#TESTS Flags
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(${file} ${testfile})
    target_link_libraries(${file} ${LINK_TEST_FLAGS})
    add_test(AllTests ${file})
endforeach(testfile ${TEST_SOURCES})

if(USE_CUDA)
    add_executable(cuda_list list.cu)
    target_compile_options(cuda_list PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(cuda_list ${LINK_TEST_FLAGS})
    add_test(AllTests cuda_list)
endif()


add_subdirectory(decoding)
add_subdirectory(mceliece)
#TODO not implemented add_subdirectory(quasicyclic)
add_subdirectory(lowweight)
# TODO currently not implemented add_subdirectory(ternary)
add_subdirectory(fq)
