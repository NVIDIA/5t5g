# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
project(5t5g LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

ADD_COMPILE_DEFINITIONS("ALLOW_EXPERIMENTAL_API")

########################################### OPTIONS ##################################################

option(ENABLE_NVTX  "Enable use of NVTX" OFF)

#####################################################################################################


############################################ DPDK ###################################################

set(DPDK_TARGET dpdk_target)
set(DPDK_TARGET_ARCH x86_64-native-linuxapp-gcc)
set(DPDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/dpdk)
message( ${DPDK_PATH})
set(DPDK_TARGET_PATH ${DPDK_PATH}/${DPDK_TARGET_ARCH})
message( ${DPDK_TARGET_PATH} )
set(DPDK_TARGET_INSTALL_PATH ${DPDK_TARGET_PATH}/install)
string(CONCAT DPDK_DISABLED_DRIVERS
        "baseband/*,"
        "bus/ifpga/*,"
        "common/cpt,"
        "common/dpaax,"
        "common/iavf,"
        "common/octeontx,"
        "common/octeontx2,"
        "crypto/nitrox,"
        "net/ark,"
        "net/atlantic,"
        "net/avp,"
        "net/axgbe,"
        "net/bnx2x,"
        "net/bnxt,"
        "net/cxgbe,"
        "net/e1000,"
        "net/ena,"
        "net/enic,"
        "net/fm10k,"
        "net/hinic,"
        "net/hns3,"
        "net/i40e,"
        "net/ixgbe,"
        "vdpa/ifc,"
        "net/igc,"
        "net/liquidio,"
        "net/netvsc,"
        "net/nfp,"
        "net/qede,"
        "net/sfc,"
        "net/thunderx,"
        "net/vdev_netvsc,"
        "net/vmxnet3,"
        )

execute_process(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND meson ${DPDK_TARGET_ARCH}
                --prefix=${DPDK_TARGET_INSTALL_PATH}
                -Dtests=false
                -Ddisable_drivers=${DPDK_DISABLED_DRIVERS}
        )

# Set RTE_PKTMBUF_HEADROOM to 0 for GPU memory best performance
execute_process(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND sed -ri "s,(RTE_PKTMBUF_HEADROOM ).*,\\10," config/rte_config.h
)

add_custom_target(${DPDK_TARGET} ALL
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}
        COMMENT Building DPDK
        COMMAND ninja install
        VERBATIM)

set(ENV{PKG_CONFIG_PATH} "${DPDK_TARGET_PATH}/meson-private")

execute_process(OUTPUT_VARIABLE DPDK_BUILD_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --cflags libdpdk.pc)

execute_process(OUTPUT_VARIABLE DPDK_LINK_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --static --libs libdpdk.pc)

set(DPDK_LIBS "-Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl ${DPDK_LINK_FLAGS}")

#####################################################################################################


############################################ CUDA ###################################################

find_package(CUDA 10 REQUIRED)
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs compat)
endif ()
mark_as_advanced(CUDA_DRIVER_LIBRARY)

##################################################################################################

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
set(COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/common/src)

add_subdirectory(receiver)
add_subdirectory(generator)

