#####################
## FindLightGBM.cmake
## Top contributors (to current version):
##   Mathias Preiner
## This file is part of the CVC4 project.
## Copyright (c) 2009-2020 by the authors listed in the file AUTHORS
## in the top-level source directory and their institutional affiliations.
## All rights reserved.  See the file COPYING in the top-level source
## directory for licensing information.
##
# Find LightGBM
# LightGBM_FOUND - system has LightGBM lib
# LightGBM_INCLUDE_DIR - the LightGBM include directory
# LightGBM_LIBRARIES - Libraries needed to use LightGBM

# find_path(LightGBM_INCLUDE_DIR NAMES lightgbm.h)
# find_library(LightGBM_LIBRARIES NAMES lightgbm )
get_filename_component(TORCHSC_ROOT "${PROJECT_SOURCE_DIR}/deps/pytorch-scatter/usr/local" ABSOLUTE)
find_library(TORCHSC_LIBRARIES NAMES torchscatter HINTS "${TORCHSC_ROOT}/lib")
message(STATUS "Found TorchScatter libs: ${TORCHSC_LIBRARIES}")
find_path(TorchScatter_INCLUDE_DIR NAMES torchscatter HINTS "${TORCHSC_ROOT}/include/")
# find_path(TorchScatter_INCLUDE_DIR NAMES include HINTS "${TORCHSC_ROOT}")
message(STATUS "Found TorchScatter libs: ${TorchScatter_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PytorchScatter
        DEFAULT_MSG
        TorchScatter_INCLUDE_DIR
        TORCHSC_LIBRARIES)

mark_as_advanced(TorchScatter_INCLUDE_DIR TORCHSC_LIBRARIES)
if(TORCHSC_LIBRARIES)
    message(STATUS "Found TorchScatter libs: ${TORCHSC_LIBRARIES}")
endif()