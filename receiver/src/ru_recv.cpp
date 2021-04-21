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

#include "ru_recv.hpp"

static struct rte_flow *setup_rules(uint16_t port_id, uint16_t vlan_tci,
								uint16_t queue_index,
								struct rte_ether_addr &ru_addr,
								uint8_t type, uint16_t flow)
{
	struct rte_flow_attr attr;
	struct rte_flow_item patterns[4];
	struct rte_flow_action actions[2];
	struct rte_flow_error err;
	struct rte_flow_action_queue queue = {.index = queue_index};
	struct rte_flow_item_eth eth_spec, eth_mask;
	struct rte_flow_item_vlan vlan_spec, vlan_mask;
	struct rte_flow_item_ecpri ecpri_spec, ecpri_mask;

	memset(&attr, 0, sizeof(attr));
	memset(patterns, 0, sizeof(patterns));
	memset(actions, 0, sizeof(actions));
	memset(&eth_spec, 0, sizeof(eth_spec));
	memset(&eth_mask, 0, sizeof(eth_mask));
	memset(&vlan_spec, 0, sizeof(vlan_spec));
	memset(&vlan_mask, 0, sizeof(vlan_mask));
	memset(&ecpri_spec, 0, sizeof(ecpri_spec));
	memset(&ecpri_mask, 0, sizeof(ecpri_mask));

	attr.ingress = 1;

	if (type == ECPRI_MSG_TYPE_IQ)
	{
		eth_spec.src.addr_bytes[0] = ru_addr.addr_bytes[0];
		eth_spec.src.addr_bytes[1] = ru_addr.addr_bytes[1];
		eth_spec.src.addr_bytes[2] = ru_addr.addr_bytes[2];
		eth_spec.src.addr_bytes[3] = ru_addr.addr_bytes[3];
		eth_spec.src.addr_bytes[4] = ru_addr.addr_bytes[4];
		eth_spec.src.addr_bytes[5] = ru_addr.addr_bytes[5];

		printf("ETH SRC %02X:%02X:%02X:%02X:%02X:%02X on queue %d\n",
				eth_spec.src.addr_bytes[0], eth_spec.src.addr_bytes[1],
				eth_spec.src.addr_bytes[2], eth_spec.src.addr_bytes[3],
				eth_spec.src.addr_bytes[4], eth_spec.src.addr_bytes[5], queue_index);

		eth_mask.src.addr_bytes[0] = 0xFF;
		eth_mask.src.addr_bytes[1] = 0xFF;
		eth_mask.src.addr_bytes[2] = 0xFF;
		eth_mask.src.addr_bytes[3] = 0xFF;
		eth_mask.src.addr_bytes[4] = 0xFF;
		eth_mask.src.addr_bytes[5] = 0xFF;

		eth_mask.dst.addr_bytes[0] = 0x0;
		eth_mask.dst.addr_bytes[1] = 0x0;
		eth_mask.dst.addr_bytes[2] = 0x0;
		eth_mask.dst.addr_bytes[3] = 0x0;
		eth_mask.dst.addr_bytes[4] = 0x0;
		eth_mask.dst.addr_bytes[5] = 0x0;

		ecpri_spec.hdr.common.type = ECPRI_MSG_TYPE_IQ;

		ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16(flow);
		ecpri_mask.hdr.type0.pc_id = 0xFFFF;
	}

	ecpri_mask.hdr.common.type = 0xFF;
	ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
	ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);
	eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
	eth_mask.type = 0xffff;

	vlan_spec.tci = rte_cpu_to_be_16(vlan_tci);
	vlan_mask.tci = rte_cpu_to_be_16(0x0fff); /* lower 12 bits only */

	vlan_spec.inner_type = rte_cpu_to_be_16(ETHER_TYPE_ECPRI);
	vlan_mask.inner_type = 0xffff;

	actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
	actions[0].conf = &queue;
	actions[1].type = RTE_FLOW_ACTION_TYPE_END;

	int pattern_idx = 0;

	patterns[pattern_idx].type = RTE_FLOW_ITEM_TYPE_ETH;
	patterns[pattern_idx].spec = &eth_spec;
	patterns[pattern_idx].mask = &eth_mask;
	pattern_idx++;

	patterns[pattern_idx].type = RTE_FLOW_ITEM_TYPE_VLAN;
	patterns[pattern_idx].spec = &vlan_spec;
	patterns[pattern_idx].mask = &vlan_mask;
	pattern_idx++;

	patterns[pattern_idx].type = RTE_FLOW_ITEM_TYPE_ECPRI;
	patterns[pattern_idx].spec = &ecpri_spec;
	patterns[pattern_idx].mask = &ecpri_mask;
	pattern_idx++;

	patterns[pattern_idx].type = RTE_FLOW_ITEM_TYPE_END;

	if (rte_flow_validate(port_id, &attr, patterns, actions, &err))
		rte_panic("Invalid flow rule: %s\n", err.message);

	return rte_flow_create(port_id, &attr, patterns, actions, &err);
}

RURecv::RURecv(int _index, struct rte_ether_addr &_eth_addr, uint16_t _ap0,
			uint16_t _ap1, uint16_t _ap2, uint16_t _ap3, uint16_t _vlan_tci,
			uint8_t _port_id, uint16_t _rxd, uint16_t _txd,
			struct rte_mempool *_mpool)
			:
			RU(_index, _eth_addr, _ap0, _ap1, _ap2, _ap3, _vlan_tci, _port_id, _rxd, _txd, _mpool)
{
	good_pkts = 0;
	bad_pkts = 0;

	CUDA_CHECK(cudaMallocHost((void **)&burst_list, MAX_BURSTS_X_PIPELINE * sizeof(struct burst_item)));

	for (int bindex = 0; bindex < MAX_BURSTS_X_PIPELINE; bindex++) {
		burst_list[bindex].bytes = 0;
		burst_list[bindex].num_mbufs = 0;
		burst_list[bindex].status = BURST_FREE;

		for (int index = 0; index < MAX_MBUFS_BURST; index++) {
			burst_list[bindex].addr[index] = 0;
			burst_list[bindex].len[index] = 0;
			burst_list[bindex].good[index] = 0;
		}
	}
	CUDA_CHECK(cudaStreamCreateWithFlags(&(stream), cudaStreamNonBlocking));
}

RURecv::~RURecv() {
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFreeHost(burst_list));

	for (int iqueue = 0; iqueue < NUM_AP; iqueue++) {
		rte_flow_destroy(port_id, frule[iqueue], NULL);
	}
}

void RURecv::setFlowRule() {
	for (int iqueue = 0; iqueue < NUM_AP; iqueue++) {
		frule[iqueue] = setup_rules(port_id, vlan_tci, rxq_list[iqueue], eth_addr, ECPRI_MSG_TYPE_IQ, eAxC_list[iqueue]);
	}
}
