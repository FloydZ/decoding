file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(testfile ${TEST_SOURCES})
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${testfile})
    string(REPLACE ".cpp" "" file ${filename})
    add_executable(ternary_${file} ${testfile})
    target_compile_options(ternary_${file} PUBLIC ${COMPILE_TEST_FLAGS})
    target_link_libraries(ternary_${file} ${LINK_TEST_FLAGS} ${TCMALLOC_LIBRARIES})
endforeach(testfile ${TEST_SOURCES})
