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

#ifndef RU_RECV_HPP_
#define RU_RECV_HPP_

#include "ru.hpp"
#include "cuda_headers.hpp"
#include "oran_receiver.hpp"
#include <rte_mbuf.h>

struct burst_item {
	uint32_t           status;
	uintptr_t          addr[MAX_MBUFS_BURST];
	uint32_t           len[MAX_MBUFS_BURST];

	struct rte_mbuf *  mbufs[MAX_MBUFS_BURST];
	uint32_t           good[MAX_MBUFS_BURST];
	int                num_mbufs;
	uint64_t           bytes;
};

class RURecv : public RU {

	public:
		RURecv(int _index, struct rte_ether_addr &_eth_addr, 
				uint16_t _ap0, uint16_t _ap1, uint16_t _ap2, uint16_t _ap3,
				uint16_t _vlan_tci, uint8_t _port_id, uint16_t _rxd, uint16_t _txd,
				struct rte_mempool * _mpool);
		~RURecv();
		void setFlowRule();

		struct burst_item * burst_list;
		cudaStream_t        stream;
		struct rte_flow *   frule[NUM_AP];
		volatile uint64_t   good_pkts;
		volatile uint64_t   bad_pkts;
};

#endif