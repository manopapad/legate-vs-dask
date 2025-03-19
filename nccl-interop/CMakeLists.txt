cmake_minimum_required(VERSION 3.22.1)

project(test_nccl VERSION 1.0 LANGUAGES CUDA CXX)

add_library(test_nccl test_nccl.cu)

find_package(legate REQUIRED)

target_link_libraries(test_nccl PUBLIC legate::legate)




// PRIVATE NCCL::NCCL











# add gnu install dirs



#   target_link_libraries(${_LEGATE_TEST_NAME}
#                         PRIVATE legate::legate
#                                 GTest::gtest
#                                 GTest::gmock
#                                 # Some test modules require delving into the private
#                                 # headers of legate, which may contain fmt headers. As a
#                                 # result, we also need to add the fmt include paths etc.
#                                 # This is not done automatically for us, since fmt is a
#                                 # PRIVATE dependency of legate (as it is not found in any
#                                 # public includes).
#                                 fmt::fmt-header-only
#                                 ${_LEGATE_TEST_ADDITIONAL_TARGETS})

#   if(Legion_USE_CUDA)
#     target_link_libraries(${_LEGATE_TEST_NAME} PRIVATE NCCL::NCCL)
#   endif()

#   rapids_test_add(NAME ${_LEGATE_TEST_NAME}
#                   COMMAND ${_LEGATE_TEST_NAME}
#                   GPUS ${_LEGATE_TEST_GPUS}
#                   PERCENT ${_LEGATE_TEST_PERCENT}
#                   INSTALL_COMPONENT_SET testing)

#   legate_install_debug_symbols(TARGET ${_LEGATE_TEST_NAME}
#                                INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

#   list(APPEND LEGATE_CONFIGURED_TESTS ${_LEGATE_TEST_NAME})
#   # propagate the updated list upwards
#   set(LEGATE_CONFIGURED_TESTS ${LEGATE_CONFIGURED_TESTS} PARENT_SCOPE)

#   foreach(src IN LISTS _LEGATE_TEST_SOURCES)
#     # Because each of the tests share a common main src file, we need to make sure we
#     # don't add a tidy target for it twice
#     list(FIND unique_src "${src}" idx)
#     if(idx EQUAL -1)
#       # not found
#       legate_add_tidy_target(SOURCE "${src}")
#       list(APPEND unique_src "${src}")
#     endif()
#   endforeach()
#   set(unique_src "${unique_src}" PARENT_SCOPE)
# endfunction()



