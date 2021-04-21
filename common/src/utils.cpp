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

#include "utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Queues
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print_rx_offloads(uint64_t offloads)
{
        uint64_t single_offload;
        int begin;
        int end;
        int bit;

        if (offloads == 0)
                return;

        begin = __builtin_ctzll(offloads);
        end = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

        single_offload = 1ULL << begin;
        for (bit = begin; bit < end; bit++) {
                if (offloads & single_offload)
                        printf(" %s",
                               rte_eth_dev_rx_offload_name(single_offload));
                single_offload <<= 1;
        }
}

void print_tx_offloads(uint64_t offloads)
{
	uint64_t single_offload;
	int begin;
	int end;
	int bit;

	if (offloads == 0)
		return;

	begin = __builtin_ctzll(offloads);
	end = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

	single_offload = 1ULL << begin;
	for (bit = begin; bit < end; bit++) {
		if (offloads & single_offload)
			printf(" %s",
			       rte_eth_dev_tx_offload_name(single_offload));
		single_offload <<= 1;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Port link
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Check the link status of all ports in up to 9s, and print them finally */
void check_all_ports_link_status(uint32_t port_mask)
{
#define CHECK_INTERVAL 100	/* 100ms */
#define MAX_CHECK_TIME 90	/* 9s (90 * 100ms) in total */
	uint16_t portid;
	uint8_t count, all_ports_up, print_flag = 0;
	struct rte_eth_link link;

	for (count = 0; count <= MAX_CHECK_TIME; count++) {
		all_ports_up = 1;
		RTE_ETH_FOREACH_DEV(portid) {
			if ((port_mask & (1 << portid)) == 0)
				continue;
			memset(&link, 0, sizeof(link));
			rte_eth_link_get_nowait(portid, &link);
			/* print link status if flag set */
			if (print_flag == 1) {
				if (link.link_status)
					printf
					    ("Port%d Link Up. Speed %u Mbps - %s\n",
					     portid, link.link_speed,
					     (link.link_duplex ==
					      ETH_LINK_FULL_DUPLEX)
					     ? ("full-duplex")
					     : ("half-duplex\n"));
				else
					printf("Port %d Link Down\n", portid);
				continue;
			}
			/* clear all_ports_up flag if any link down */
			if (link.link_status == ETH_LINK_DOWN) {
				all_ports_up = 0;
				break;
			}
		}
		/* after finally printing all link status, get out */
		if (print_flag == 1)
			break;

		if (all_ports_up == 0) {
			//printf(".");
			fflush(stdout);
			rte_delay_ms(CHECK_INTERVAL);
		}

		/* set the print_flag if all ports up or timeout */
		if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
			print_flag = 1;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Input args
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void usage(const char *prgname)
{
	printf("\n\n%s [EAL options] -- g|h|n\n"
	       " -g GPU DEVICE: GPU device ID\n"
	       " -h HELP\n"
		   " -n CUDA PROFILER: Enable CUDA profiler with NVTX for nvvp\n"
		   ,
	       prgname);
}

uint64_t get_timestamp_ns(void)
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
	{
    	fprintf(stderr, "clock_gettime failed\n");
		return 0;
    }
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_timestamp_ns() + ns, start_t = 0;
    while ((start_t = get_timestamp_ns()) < end_t) {
        for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
            __asm__ __volatile__ ("");
        }
    }
}