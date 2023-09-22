# Link libjpeg library for all executables
link_libraries(jpeg)

# Enable programs to include utils.cpp & utils.hpp with no directory prefix
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

message("CMAKE_CXX_FLAGS Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS Release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_subdirectory(cpu)
add_subdirectory(gpu)
