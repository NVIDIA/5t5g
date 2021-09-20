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

#ifndef RU_HPP_
#define RU_HPP_

#include "constants.hpp"
#include "utils.hpp"

extern struct rte_ether_addr ru0_addr;
extern uint16_t ru0_ap[NUM_AP];
extern uint16_t ru0_vlan;
extern struct rte_ether_addr ru1_addr;
extern uint16_t ru1_ap[NUM_AP];
extern uint16_t ru1_vlan;

class RU {

    public:
        RU(int _index, struct rte_ether_addr &_eth_addr,
                uint16_t _ap0, uint16_t _ap1, uint16_t _ap2, uint16_t _ap3,
                uint16_t _vlan_tci, uint8_t _port_id, uint16_t _rxd, uint16_t _txd,
                struct rte_mempool * _mpool);
        ~RU();
        void setupQueues();

        int                     index;
        char                    name[RU_NAME_LEN];
        uint16_t                vlan_tci;
        uint16_t                rxd;
        uint16_t                txd;
        uint8_t                 port_id;
        uint8_t                 rxq_list[NUM_AP];
        uint8_t                 txq_list[NUM_AP];
        struct rte_ether_addr   eth_addr;
        uint16_t                eAxC_list[NUM_AP];
        struct rte_mempool *    mpool;
};

#endif