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

#ifndef RU_GEN_HPP_
#define RU_GEN_HPP_

#include "ru.hpp"
#include "oran_generator.hpp"

struct pkt_hdr_template
{
    struct rte_ether_hdr  eth;
    struct rte_vlan_hdr   vlan;
    struct oran_ecpri_hdr ecpri;
} __attribute__((packed));

class RUGen : public RU {

    public:
        RUGen(int _index, struct rte_ether_addr &_eth_addr, 
                uint16_t _ap0, uint16_t _ap1, uint16_t _ap2, uint16_t _ap3, 
                uint16_t _vlan_tci, uint8_t _port_id,  uint16_t _rxd, uint16_t _txd,
                struct rte_mempool * _mpool, struct rte_ether_addr &_dst_eth_addr, int _mu, int _tx_offset_pkts_ns, int _tx_interval_pkts)
                    :  
                    RU(_index, _eth_addr, _ap0, _ap1, _ap2, _ap3, _vlan_tci, _port_id, _rxd, _txd, _mpool)
    {
             rte_ether_addr_copy(&_dst_eth_addr, &dst_eth_addr);

            for(int ihdr=0; ihdr < NUM_AP; ihdr++)
            {
                rte_ether_addr_copy(&dst_eth_addr, &pkt_hdr[ihdr].eth.d_addr);
                rte_ether_addr_copy(&eth_addr, &pkt_hdr[ihdr].eth.s_addr);
                pkt_hdr[ihdr].eth.ether_type 			= rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
                pkt_hdr[ihdr].vlan.vlan_tci  			= rte_cpu_to_be_16(vlan_tci);
                pkt_hdr[ihdr].vlan.eth_proto 			= rte_cpu_to_be_16(ETHER_TYPE_ECPRI);
                pkt_hdr[ihdr].ecpri.ecpriVersion       	= ORAN_DEF_ECPRI_VERSION;
                pkt_hdr[ihdr].ecpri.ecpriReserved      	= ORAN_DEF_ECPRI_RESERVED;
                pkt_hdr[ihdr].ecpri.ecpriConcatenation 	= ORAN_ECPRI_CONCATENATION_NO;
                pkt_hdr[ihdr].ecpri.ecpriPcid          	= rte_cpu_to_be_16(eAxC_list[ihdr]);
                pkt_hdr[ihdr].ecpri.ecpriMessage       	= ECPRI_MSG_TYPE_IQ;
                pkt_hdr[ihdr].ecpri.ecpriEbit     	 	= 1;
                pkt_hdr[ihdr].ecpri.ecpriSubSeqid 	 	= 0;
            }

            mu = _mu;
            tx_offset_pkts_ns = _tx_offset_pkts_ns;
            tx_interval_pkts = _tx_interval_pkts; //128;

            if(mu == 0) //1ms 15kHZ SCS
            {
                tx_interval_step = 10;
                tx_interval_ns = tx_interval_step * 100 * 1000;
                tx_interval_s = (float)tx_interval_step * 0.0001;
            }
            else //by default 500us 30kHZ SCS
            {
                tx_interval_step = 5;
                tx_interval_ns = tx_interval_step * 100 * 1000;
                tx_interval_s = (float)tx_interval_step * 0.0001;
            }
        }

        ~RUGen();

        int                     mu;
        int                     tx_interval_step;
        int                     tx_interval_ns;
        int                     tx_offset_pkts_ns;
        float                   tx_interval_s;
        int                     tx_interval_pkts;
        pkt_hdr_template        pkt_hdr[NUM_AP];
        struct rte_ether_addr   dst_eth_addr;        
};

#endif