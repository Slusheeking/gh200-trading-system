# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/gh200-trading-system/_deps/pybind11-src")
  file(MAKE_DIRECTORY "/home/ubuntu/gh200-trading-system/_deps/pybind11-src")
endif()
file(MAKE_DIRECTORY
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-build"
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix"
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/tmp"
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/src"
  "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/gh200-trading-system/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
