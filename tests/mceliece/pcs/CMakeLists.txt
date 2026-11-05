file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(mceliece_test_pcs_${file} ${testfile})
    target_compile_options(mceliece_test_pcs_${file} PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(mceliece_test_pcs_${file} ${LINK_TEST_FLAGS} ${TCMALLOC_LIBRARIES})
endforeach(testfile ${TEST_SOURCES})
