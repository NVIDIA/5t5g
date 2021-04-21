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
#include "utils.hpp"
#include "ru_recv.hpp"

////////////////////////////////////////////////////////////////////////
//// Command line config params
////////////////////////////////////////////////////////////////////////
static uint32_t conf_enabled_port_mask  = 0;	//mask of enabled ports
static int conf_gpu_device_id           = 0;
static int conf_port_id                 = 0;
static uint32_t conf_data_room_size     = DEF_DATA_ROOM_SIZE;
static int conf_pkt_burst_size          = MAX_MBUFS_BURST;
static int conf_num_pipelines           = NUM_RU;
static int conf_nvprofiler              = 0;
static int conf_num_rx_queue            = NUM_AP*NUM_RU;
static int conf_num_tx_queue            = NUM_AP*NUM_RU;
static int conf_mempool_cache           = RTE_MEMPOOL_CACHE_MAX_SIZE;
static int conf_nb_mbufs                = DEF_NB_MBUF;
static int conf_nb_rxd                  = DEF_RX_DESC;
static int conf_nb_txd                  = DEF_TX_DESC;

static const char short_options[] = 
    "g:" /* GPU device */
    "h"	 /* help */
    "n:" /* NVTX Profiler */
    ;

////////////////////////////////////////////////////////////////////////
//// DPDK config
////////////////////////////////////////////////////////////////////////
struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
struct rte_mempool *mpool_payload;
struct rte_pktmbuf_extmem ext_mem;

static struct rte_eth_conf port_eth_conf = {
    .rxmode = {
           .mq_mode = ETH_MQ_RX_RSS,
           .max_rx_pkt_len = conf_data_room_size,
           .split_hdr_size = 0,
           .offloads = 0,
           },
    .txmode = {
           .mq_mode = ETH_MQ_TX_NONE,
           .offloads = 0,
           },
    .rx_adv_conf = {
            .rss_conf = {
                     .rss_key = NULL,
                     .rss_hf = ETH_RSS_IP
                    },
            },
};

////////////////////////////////////////////////////////////////////////
//// Inter-threads communication
////////////////////////////////////////////////////////////////////////
volatile bool force_quit;

////////////////////////////////////////////////////////////////////////
//// RU Objects
////////////////////////////////////////////////////////////////////////
RURecv * ru0;
RURecv * ru1;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Signal Handler
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM || signum == SIGUSR1) {
        printf("\n\nSignal %d received, preparing to exit...\n", signum);
        ACCESS_ONCE(force_quit) = 1;
    }
}

static void print_stats(void)
{
    struct rte_eth_stats stats;
    int index_queue = 0;

    rte_eth_stats_get(conf_port_id, &stats);

    fflush(stdout);
    printf("RX QUEUES:\n");
    for (index_queue = 0; index_queue < conf_num_rx_queue; index_queue++)
    {
        printf("\t\tQueue %d: packets = %ld bytes = %ld dropped = %ld\n",
             index_queue, stats.q_ipackets[index_queue],
             stats.q_ibytes[index_queue], stats.q_errors[index_queue]);
    }

    printf("\nTot RX packets: %lu, Tot Rx bytes: %ld\n", stats.ipackets, stats.ibytes);

    printf("\nERRORS:\n");
    printf("Total of RX packets dropped by the HW, because there are no available buffer (i.e. RX queues are full)=%" PRIu64 "\n", stats.imissed);
    printf("Total number of erroneous received packets=%" PRIu64 "\n", stats.ierrors);
    printf("Total number of RX mbuf allocation failures=%" PRIu64 "\n", stats.rx_nombuf);
    printf("\n====================================================\n");
    fflush(stdout);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Input args
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned int conf_parse_device_id(const char *q_arg)
{
    char *end = NULL;
    unsigned long n;

    n = strtoul(q_arg, &end, 10);
    if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
        return 0;

    return n;
}

static int parse_args(int argc, char **argv)
{
    int opt, ret = 0;
    char **argvopt;
    int option_index;
    char *prgname = argv[0];
    int totDevs;
    cudaError_t cuda_ret = cudaSuccess;
    argvopt = argv;

    while ((opt = getopt_long(argc, argvopt, short_options, NULL, &option_index)) != EOF)
    {
        switch (opt) {

        case 'g':
            conf_gpu_device_id = conf_parse_device_id(optarg);
            break;

        case 'h':
            usage(prgname);
            return -2;
        case 'n':
            conf_nvprofiler = 1;
            break;

        case 0:
            break;

        default:
            usage(prgname);
            return -1;
        }
    }

    if (optind >= 0)
        argv[optind - 1] = prgname;

    ret = optind - 1;
    optind = 1;

    if (totDevs < conf_gpu_device_id) {
        fprintf(stderr, "Erroneous GPU device ID (%d). Tot GPUs: %d\n", conf_gpu_device_id, totDevs);
        return -1;
    }

    if ( ((conf_num_pipelines * 2) + 1) > (int)rte_lcore_count()) {
        fprintf(stderr,
            "Required conf_num_pipelines+1 (%d), cores launched=(%d)\n",
            conf_num_pipelines + 1, rte_lcore_count());
        return -1;
    }

    return ret;
}

static int validator_core(void *arg)
{
    long pipeline_idx = (long)arg;
    RURecv * ru;
    cudaStream_t stream;
    int nb_tx = 0, bindex = 0, ipkt = 0, opkt = 0;
    struct burst_item * blist;
    struct rte_mbuf * tx_mbufs[MAX_MBUFS_BURST];
    uint64_t start_tx = get_timestamp_ns();
    int itxq = 0;

    if(pipeline_idx == 0)
        ru = ru0;
    else
        ru = ru1;

    blist = ru->burst_list;
    stream = ru->stream;

    printf("\n=======> VALIDATOR CORE %d on RU %ld: %s\n", rte_lcore_id(), pipeline_idx, ru->name);

    while(ACCESS_ONCE(force_quit) == 0)
    {
        PUSH_RANGE("wait_burst", 7);
        while(ACCESS_ONCE(force_quit) == 0 && ACCESS_ONCE(blist[bindex].status) != BURST_DONE);
        rte_rmb();
        POP_RANGE;

        for(ipkt = 0, opkt = 0; ipkt < blist[bindex].num_mbufs && opkt < blist[bindex].num_mbufs; ipkt++)
        {
            if(blist[bindex].good[ipkt] == 1)
                opkt++;
            //Set this value to some random number to prove the kernel actually changed it to 0 or 1
            ACCESS_ONCE(blist[bindex].good[ipkt]) = 2;

            rte_pktmbuf_free(blist[bindex].mbufs[ipkt]);
        }

        ru->good_pkts += opkt;
        ru->bad_pkts += blist[bindex].num_mbufs - opkt;

        ACCESS_ONCE(blist[bindex].num_mbufs) 	= 0;
        ACCESS_ONCE(blist[bindex].bytes)		= 0;
        ACCESS_ONCE(blist[bindex].status)		= BURST_FREE;
        rte_mb();

        bindex = (bindex+1) % MAX_BURSTS_X_PIPELINE;
    }

    return 0;
}

static int rx_core(void *arg)
{
    long pipeline_idx = (long)arg;
    RURecv * ru;
    cudaStream_t stream;
    int nb_rx = 0, bindex = 0;
    struct burst_item * blist;
    int irxq = 0;

    if(pipeline_idx == 0)
        ru = ru0;
    else
        ru = ru1;
    
    blist = ru->burst_list;
    stream = ru->stream;

    printf("\n=======> RX CORE %d on RU %ld: %s\n", rte_lcore_id(), pipeline_idx, ru->name);
    
    while (ACCESS_ONCE(force_quit) == 0)
    {
        if(ACCESS_ONCE(blist[bindex].status) != BURST_FREE)
        {
            fprintf(stderr, "Burst %d is not free. Pipeline it's too slow, quitting...\n", bindex);
            ACCESS_ONCE(force_quit) = 1;
            return -1;
        }

        PUSH_RANGE("rx_burst", 1);
        nb_rx = 0;
        while (ACCESS_ONCE(force_quit) == 0 && nb_rx < (conf_pkt_burst_size - GAP_PKTS))
        {
            nb_rx += rte_eth_rx_burst(conf_port_id, ru->rxq_list[irxq], 
                                        &(blist[bindex].mbufs[nb_rx]),
                                        (conf_pkt_burst_size - nb_rx)
                                    );
            irxq = (irxq+1)%NUM_AP;
        }

        POP_RANGE;

        if (!nb_rx)
            continue;

        PUSH_RANGE("prep_pkts", 3);

        blist[bindex].num_mbufs = nb_rx;
        for(int index=0; index < nb_rx; index++)
        {
            blist[bindex].addr[index] 	= (uintptr_t) rte_pktmbuf_mtod_offset(blist[bindex].mbufs[index], void*, 0);
            blist[bindex].len[index] 	= blist[bindex].mbufs[index]->data_len;
            blist[bindex].bytes 		+= blist[bindex].mbufs[index]->pkt_len;
        }
        rte_wmb();

        POP_RANGE;

        ACCESS_ONCE(blist[bindex].status) = BURST_READY;
        rte_wmb();

        PUSH_RANGE("macswap_gpu", 4);
        launch_gpu_processing(
                            blist[bindex].addr, blist[bindex].num_mbufs, blist[bindex].good, &(blist[bindex].status),
                            ru->eAxC_list[0], ru->eAxC_list[1], ru->eAxC_list[2], ru->eAxC_list[3], 
                            CUDA_BLOCKS, THREADS_BLOCK, stream
                        );
        POP_RANGE;

        bindex = (bindex+1) % MAX_BURSTS_X_PIPELINE;
    }

    return 0;
}

static int stats_core(void* arg)
{
    uint64_t sec = 0;
    uint64_t start = get_timestamp_ns();

    while(ACCESS_ONCE(force_quit) == 0)
    {
        while((get_timestamp_ns() - start) < 1 * 1000 * 1000 * 1000);

        fprintf(stderr, "%ld sec) %s OK=%ld ERR=%ld / %s OK=%ld ERR=%ld\n", 
                    sec,
                    ru0->name, ru0->good_pkts, ru0->bad_pkts,
                    ru1->name, ru1->good_pkts, ru1->bad_pkts
                    );

        start = get_timestamp_ns();
        sec++;
    }

    return 0;
}


int main(int argc, char **argv)
{
    struct rte_eth_dev_info dev_info;
    uint8_t socketid;
    int ret = 0, index_q, index_queue = 0, secondary_id = 0;
    uint16_t nb_ports;
    unsigned lcore_id;
    long icore = 0;
    uint16_t nb_rxd = DEF_RX_DESC;
    uint16_t nb_txd = DEF_TX_DESC;
    struct rte_flow_error flowerr;

    //Prevent any useless profiling
    cudaProfilerStop();

    printf("************ 5T for 5G Receiver ************\n\n");

    /* ================ PARSE ARGS ================ */
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
    argc -= ret;
    argv += ret;

    /* parse application arguments (after the EAL ones) */
    ret = parse_args(argc, argv);
    if (ret == -2)
        rte_exit(EXIT_SUCCESS, "\n");
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid 5T for 5G Receiver arguments\n");

    /* ================ FORCE QUIT HANDLER ================ */
    ACCESS_ONCE(force_quit) = 0;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGUSR1, signal_handler);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// GPU/NIC devices setup
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaSetDevice(conf_gpu_device_id);
    cudaFree(0);

    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0)
        rte_exit(EXIT_FAILURE, "No Ethernet ports - bye\n");

    rte_eth_dev_info_get(conf_port_id, &dev_info);
    printf("\nDevice driver name in use: %s... \n", dev_info.driver_name);

    if (strcmp(dev_info.driver_name, "mlx5_pci") != 0)
        rte_exit(EXIT_FAILURE, "Non-Mellanox NICs have not been validated with 5g5t\n");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// GPU MEMORY MEMPOOL
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ext_mem.elt_size = conf_data_room_size + RTE_PKTMBUF_HEADROOM;
    ext_mem.buf_len = RTE_ALIGN_CEIL(conf_nb_mbufs * ext_mem.elt_size, GPU_PAGE_SIZE);
    ext_mem.buf_iova = RTE_BAD_IOVA;

    CUDA_CHECK(cudaMalloc(&ext_mem.buf_ptr, ext_mem.buf_len));
    if (ext_mem.buf_ptr == NULL)
        rte_exit(EXIT_FAILURE, "Could not allocate GPU memory\n");

    unsigned int flag = 1;
    CUresult status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ext_mem.buf_ptr);
    if (CUDA_SUCCESS != status) {
        rte_exit(EXIT_FAILURE, "Could not set SYNC MEMOP attribute for GPU memory at %llx\n", (CUdeviceptr)ext_mem.buf_ptr);
    }
    ret = rte_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova, GPU_PAGE_SIZE);
    if (ret)
        rte_exit(EXIT_FAILURE, "Could not register GPU memory\n");

    ret = rte_dev_dma_map(rte_eth_devices[conf_port_id].device, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
    if (ret)
        rte_exit(EXIT_FAILURE, "Could not DMA map EXT memory\n");
    mpool_payload = rte_pktmbuf_pool_create_extbuf("payload_mpool", conf_nb_mbufs,
                                            conf_mempool_cache, 0, ext_mem.elt_size, 
                                            rte_socket_id(), &ext_mem, 1);
    if (mpool_payload == NULL)
        rte_exit(EXIT_FAILURE, "Could not create EXT memory mempool\n");
    
    port_eth_conf.rxmode.offloads = DEV_RX_OFFLOAD_JUMBO_FRAME;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// PORT SETUP
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Initializing port %u with %d RX queues and %d TX queues...\n", conf_port_id, conf_num_rx_queue, conf_num_tx_queue);

    ret = rte_eth_dev_configure(conf_port_id, conf_num_rx_queue, conf_num_tx_queue, &port_eth_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
             "Cannot configure device: err=%d, port=%u\n", ret,
             conf_port_id);

    printf("Port RX offloads: ");
    print_rx_offloads(port_eth_conf.rxmode.offloads);
    printf("\n");
    printf("Port TX offloads: ");
    print_tx_offloads(port_eth_conf.txmode.offloads);
    printf("\n");
    
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(conf_port_id, &nb_rxd, &nb_txd);
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
             "Cannot adjust number of descriptors: err=%d, port=%u\n",
             ret, conf_port_id);

    rte_eth_macaddr_get(conf_port_id, &conf_ports_eth_addr[conf_port_id]);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Configure RUs
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ru0 = new RURecv(0, ru0_addr, ru0_ap[0], ru0_ap[1], ru0_ap[2], ru0_ap[3], ru0_vlan, conf_port_id, nb_rxd, nb_txd, mpool_payload);
    ru0->setupQueues();
    
    ru1 = new RURecv(1, ru1_addr, ru1_ap[0], ru1_ap[1], ru1_ap[2], ru1_ap[3], ru1_vlan, conf_port_id, nb_rxd, nb_txd, mpool_payload);
    ru1->setupQueues();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// START DEVICE
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (rte_flow_isolate(conf_port_id, 1, &flowerr)) {
        rte_panic("Flow isolation failed: %s\n", flowerr.message);
    }

    ret = rte_eth_dev_start(conf_port_id);
    if (ret != 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n", ret, conf_port_id);
    
    ru0->setFlowRule();
    ru1->setFlowRule();

    printf("Port %d, MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
           conf_port_id,
           conf_ports_eth_addr[conf_port_id].addr_bytes[0],
           conf_ports_eth_addr[conf_port_id].addr_bytes[1],
           conf_ports_eth_addr[conf_port_id].addr_bytes[2],
           conf_ports_eth_addr[conf_port_id].addr_bytes[3],
           conf_ports_eth_addr[conf_port_id].addr_bytes[4],
           conf_ports_eth_addr[conf_port_id].addr_bytes[5]
        );

    check_all_ports_link_status(conf_enabled_port_mask);

    if (conf_nvprofiler)
        cudaProfilerStart();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// START RX/TX CORES
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    PUSH_RANGE("launch_statscore", 5);
    secondary_id = rte_get_next_lcore(secondary_id, 1, 0);
    rte_eal_remote_launch(stats_core, nullptr, secondary_id);
    POP_RANGE;

    for(icore = 0; icore < conf_num_pipelines; icore++)
    {
        PUSH_RANGE("launch_validatorcore", 5);
        secondary_id = rte_get_next_lcore(secondary_id, 1, 0);
        rte_eal_remote_launch(validator_core, (void *)icore, secondary_id);
        POP_RANGE;

        PUSH_RANGE("launch_rxcore", 6);
        secondary_id = rte_get_next_lcore(secondary_id, 1, 0);
        rte_eal_remote_launch(rx_core, (void *)icore, secondary_id);
        POP_RANGE;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// WAIT RX/TX CORES
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    icore = 0;
    RTE_LCORE_FOREACH_WORKER(icore) {
        if (rte_eal_wait_lcore(icore) < 0) {
            fprintf(stderr, "bad exit for coreid: %ld\n",
                icore);
            break;
        }
    }

    if (conf_nvprofiler)
        cudaProfilerStop();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Print stats
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print_stats();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Close network device
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Closing port %d...", conf_port_id);
    rte_eth_dev_stop(conf_port_id);
    rte_eth_dev_close(conf_port_id);
    printf(" Done. Bye!\n");

    return 0;
}
