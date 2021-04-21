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

#include "ru.hpp"

struct rte_ether_addr ru0_addr = {
    .addr_bytes = {0x22, 0x06, 0x9C, 0x9A, 0x25, 0xA5}
};

uint16_t ru0_ap[NUM_AP] = {0, 4, 6, 9};
uint16_t ru0_vlan = 0;

struct rte_ether_addr ru1_addr = {
    .addr_bytes = {0x24, 0x42, 0xA1, 0xD1, 0xC1, 0xA7}
};
uint16_t ru1_ap[NUM_AP] = {1, 3, 5, 7};
uint16_t ru1_vlan = 0;

RU::RU(int _index, struct rte_ether_addr &_eth_addr, 
        uint16_t _ap0, uint16_t _ap1, uint16_t _ap2, uint16_t _ap3,
        uint16_t _vlan_tci, uint8_t _port_id, uint16_t _rxd, uint16_t _txd,
        struct rte_mempool * _mpool)
{
    index = _index;
    vlan_tci = _vlan_tci;
    port_id = _port_id;
    rxd = _rxd;
    txd = _txd;
    mpool = _mpool;

    snprintf(name, RU_NAME_LEN, "RU #%d", index);

    rte_ether_addr_copy(&_eth_addr, &eth_addr);
    
    for(int iloop=0; iloop < NUM_AP; iloop++)
    {
        rxq_list[iloop] = (NUM_AP * index) + iloop;
        txq_list[iloop] = rxq_list[iloop];
    }

    eAxC_list[0] = _ap0;
    eAxC_list[1] = _ap1;
    eAxC_list[2] = _ap2;
    eAxC_list[3] = _ap3;
}

RU::~RU() {}

void RU::setupQueues() {
    int ret = 0;
    uint8_t socketid = (uint8_t) rte_lcore_to_socket_id(rte_lcore_id());

    for(int iqueue = 0; iqueue < NUM_AP; iqueue++) {
        ret = rte_eth_rx_queue_setup(port_id, rxq_list[iqueue], rxd, socketid, NULL, mpool);
        if (ret < 0)
            rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%u\n", ret, port_id);

        ret = rte_eth_tx_queue_setup(port_id, txq_list[iqueue], txd, socketid, NULL);
        if (ret < 0)
            rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup: err=%d, port=%u\n", ret, port_id);
    }
}
