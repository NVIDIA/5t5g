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

#ifndef COSTANTS_H
#define COSTANTS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#define MAX(a, b) ((a) > (b) ? a : b)
#define MIN(a, b) ((a) < (b) ? a : b)

static constexpr uint32_t BURST_FREE = 0;
static constexpr uint32_t BURST_READY = 1;
static constexpr uint32_t BURST_DONE = 2;
static constexpr uint32_t BURST_EXIT = 3;

static constexpr uint32_t DEF_DATA_ROOM_SIZE = 1024;
static constexpr uint32_t GAP_PKTS = 4;

static constexpr uint32_t GPU_PAGE_SHIFT = 16;
static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);

static constexpr uint32_t RU_NAME_LEN = 128;
static constexpr uint32_t NUM_RU = 2;
static constexpr uint32_t NUM_AP = 4; // Warning: do not change this number!
static constexpr uint32_t DEF_RX_DESC = 1024;
static constexpr uint32_t DEF_TX_DESC = 1024;
static constexpr uint32_t DEF_NB_MBUF = 16384;
static constexpr uint8_t  PRBS_PER_PACKET = 16;
static constexpr uint8_t  IQ_SAMPLE_SIZE = 16;

static constexpr uint32_t MAX_BURSTS_X_PIPELINE = 4096;
static constexpr uint32_t MAX_MBUFS_BURST = 512;
static constexpr uint32_t THREADS_BLOCK = MAX_MBUFS_BURST;
static constexpr uint32_t CUDA_BLOCKS = (MAX_MBUFS_BURST + THREADS_BLOCK - 1) / THREADS_BLOCK;

#endif