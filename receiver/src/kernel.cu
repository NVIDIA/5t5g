/* 
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "constants.hpp"
#include "cuda_headers.hpp"
#include "oran_receiver.hpp"

//#define DEBUG_PRINT

__global__ void kernel_check_ecpri_flow(uintptr_t * addr, int num_pkts, uint32_t * good, 
                                    uint32_t * status, uint16_t ap0, uint16_t ap1, uint16_t ap2, uint16_t ap3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t flow_id;

    if (idx < num_pkts) {
        flow_id = oran_umsg_get_flowid((((uint8_t *) (addr[idx]))));
        if(
            flow_id != ap0 &&  flow_id != ap1 && 
            flow_id != ap2 &&  flow_id != ap3
        )
            good[idx] = 0;
        else
            good[idx] = 1;

#ifdef DEBUG_PRINT
		printf("flowId: %d, ap0: %d, ap1: %d, ap2: %d, ap3: %d\n", flowId, ap0, ap1, ap2, ap3);
#endif

	}
	__syncthreads();

	if (idx == 0) {
		status[0] = BURST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void launch_gpu_processing(uintptr_t * addr, int num_pkts, uint32_t * good, uint32_t * status,
                            uint16_t ap0, uint16_t ap1, uint16_t ap2, uint16_t ap3,
                            int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
    assert(cuda_blocks == 1);
    assert(cuda_threads > 0);

    if (addr == NULL)
        return;

    CUDA_CHECK(cudaGetLastError());
    kernel_check_ecpri_flow<<<cuda_blocks, cuda_threads, 0, stream >>>(addr, num_pkts, good, status, ap0,  ap1, ap2, ap3);
    CUDA_CHECK(cudaGetLastError());
}