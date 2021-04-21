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

set(TARGET 5t5g_receiver)

set(5T5G_RECV_SRCS
        ${COMMON_SRC}/utils.cpp
        ${COMMON_SRC}/ru.cpp
        src/kernel.cu
        src/main.cpp
        src/ru_recv.cpp
        )

add_executable(${TARGET} ${5T5G_RECV_SRCS})

# Link with nvToolsExt
if (ENABLE_NVTX)
    set(NVTX_LIB nvToolsExt)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

target_include_directories(${TARGET}    PUBLIC include 
                                        PUBLIC ${COMMON_INCLUDE}
                                        PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries(${TARGET} ${DPDK_LIBS} ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${NVTX_LIB})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -std=c++11  ${CMAKE_COMMON_FLAGS} -g3")

# Enable to debug CUDA kernels
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS} -G -lineinfo")

set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
add_dependencies(${TARGET} ${DPDK_TARGET})

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/5t5g)