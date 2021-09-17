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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <netinet/in.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <array>

///////////////////////////////////////////////////////////////////////////
//// DPDK headers
///////////////////////////////////////////////////////////////////////////
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_flow.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

/////////////////////////////////////////////////////////////////
// Command Line Options
/////////////////////////////////////////////////////////////////
void print_rx_offloads(uint64_t offloads);
void print_tx_offloads(uint64_t offloads);
void check_all_ports_link_status(uint32_t port_mask);
void usage(const char *prgname);

/////////////////////////////////////////////////////////////////
// Time
/////////////////////////////////////////////////////////////////
uint64_t get_timestamp_ns(void);
void wait_ns(uint64_t ns);

#endif