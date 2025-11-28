cmake_minimum_required(VERSION 3.15)
project(TEST)
find_package(PCL 1.15 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(include)

add_executable(TEST src/main.cpp src/Kmeans.cpp)

target_link_libraries(TEST ${PCL_LIBRARIES})

target_compile_definitions(TEST PRIVATE
    _CRT_SECURE_NO_WARNINGS
)