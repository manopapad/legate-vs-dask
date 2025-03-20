cmake_minimum_required(VERSION 3.22.1)

project(test_nccl VERSION 1.0 LANGUAGES CUDA CXX)

add_library(test_nccl test_nccl.cu)

find_package(legate REQUIRED)

find_path(NCCL_HEADER NAMES nccl.h)
find_library(NCCL_LIB NAMES nccl)

if(NCCL_HEADER AND NCCL_LIB)
  add_library(NCCL STATIC IMPORTED)
  add_library(NCCL::NCCL ALIAS NCCL)
  set_target_properties(NCCL PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_HEADER}")
else()
  message(FATAL_ERROR "NCCL not found")
endif()

target_link_libraries(test_nccl PUBLIC legate::legate PRIVATE NCCL::NCCL)
