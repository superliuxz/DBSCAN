cmake_minimum_required(VERSION 3.15)
project(Pagerank)

set(CMAKE_CXX_STANDARD 17)

add_executable(Pagerank main.cpp include/matrix.h)

##
# https://stackoverflow.com/questions/49095153/cmake-failing-to-link-against-google-profiler-gperftools
##
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
find_package(Gperftools QUIET REQUIRED)
if (Gperftools_FOUND)
    MESSAGE("Gperftools found: ${GPERFTOOLS_LIBRARIES} ${GPERFTOOLS_INCLUDE_DIR}")
    target_link_libraries(Pagerank ${GPERFTOOLS_LIBRARIES})
    include_directories("${GPERFTOOLS_INCLUDE_DIR}")
endif ()
