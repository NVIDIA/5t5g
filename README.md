# 5T for 5G

The goal of this project is to provide an example on how to use [5T for 5G features](https://news.developer.nvidia.com/new-real-time-smartnic-technology-5t-for-5g) through the DPDK library. To understand the terminology and acronyms, please refer to the [ORAN standard](https://www.o-ran.org)):

5T for 5G constains two applications: 
* Generator: simulates two O-RUs sending O-RAN formatted U-plane packets using different slot durations
* Receiver: simulates an O-DU which is capable to receive O-RAN CUS U-plane packets and distinguish the data flows coming from the two O-RUs

Please note that main goal of these applications is not to showcase performance, but to demonstrate how to enable 5T for 5G features in you application.

### Generator

C++ application simulating two O-RUs: one working at 30 kHz SCS (500 us of slot duration) and the other working at 15 kHz SCS (1 ms slot duration). Each RU sends O-RAN formatted U-plane packets from CPU memory using Accurate Send Scheduling (slot start timestamp + 700 us).
Each O-RU has its own MAC address and it simulates traffic coming from 4 different antenna ports (or eAxC IDs).

### Receiver

C++ application simulating an O-DU able to receive O-RAN CUS U-plane packet into GPU memory (GPUDirect RDMA) using DPDK flow steering rules to distinguish traffic coming from the two different O-RUs. Specifically the following rules are applied to distinguish different flows:
* MAC address
* VLAN tag
* O-RAN message type (C/U-plane `ecpriMessage`)
* Antenna port ID (`eAxC ID` or `ecpriPcid`)

Each flow has a dedicated DPDK RX queue which means that the receiver has 8 different RX queues (2 O-RUs with 4 antenna ports).
These RX queues stores packets in a DPDK mempool which resides in GPU memory and it's created by the following list of functions:
```
cudaMalloc();
rte_extmem_register();
rte_dev_dma_map();
rte_pktmbuf_pool_create_extbuf();
```

**Note** The receiver requires GPUDirect RDMA enabled on the system.


## Prerequisites

1. CMake (version 3.10 or newer)

   If you have a version of CMake installed, the version number can be determined as follows:
   ```
   cmake --version
   ```
   If you do not have CMake installed, or if the version is older than 3.10, binary distributions
   can be obtained from https://cmake.org/download/. To install in a non-system directory,
   specify a local path for the `--prefix` argument to the installation script.

   ```
   wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh
   ./cmake-3.13.2-Linux-x86_64.sh --skip-license --prefix=/home/username/usr/local
   ```

2. CUDA (version 10 or newer)

   CMake intrinsic CUDA support (available in CMake 3.8 or newer) will automatically detect
   a CUDA installation using a CUDA compiler (`nvcc`), which is  located via the `PATH` environment
   variable. To check for `nvcc` in your `PATH`:
   
   ```
   which nvcc
   ```

   To use a non-standard CUDA installation path (or to use a specific version of CUDA):
   
   ```
   export PATH=/usr/local/cuda-11/bin:$PATH
   ```

   For more information on CUDA support in CMake, see https://devblogs.nvidia.com/building-cuda-applications-cmake/.

3. Mellanox OFED (version 5.4-1.0.3.0 or newer)

   The following instructions are for installing MOFED 5.4-1.0.3.0 for Ubuntu 18.04:
   ```
   export OFED_VERSION=5.4-1.0.3.0
   export UBUNTU_VERSION=18.04

   wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64.tgz
   tar xvf MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64.tgz
   cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64
   sudo ./mlnxofedinstall --upstream-libs --dpdk --with-mft
   sudo /etc/init.d/openibd restart
   ```

4. Mellanox ConnectX-6 Dx NIC

   Installing MOFED 5.4-1.0.3.0 should automatically upgrade ConnectX-6 Dx firmware to version 22.31.1014. If FW version on the card is older than 22.31.1014, please upgrade manually.

   A number of FW configs need to be changes in order to enable 5T for 5G.

   Native eCPRI flow steering enable:
   ```
   sudo mlxconfig -d <NIC Bus Id> s PROG_PARSE_GRAPH=1
   sudo mlxconfig -d <NIC Bus Id> s FLEX_PARSER_PROFILE_ENABLE=4
   ```

   ConnectX-6 Dx offload for accurate send scheduling enable:
   ```
   sudo mlxconfig -d <NIC Bus Id> s REAL_TIME_CLOCK_ENABLE=1
   sudo mlxconfig -d <NIC Bus Id> s ACCURATE_TX_SCHEDULER=1
   ```

   Please reser the NIC FW for the changes to take effect:
   ```
   sudo mlxfwreset -d <NIC Bus Id> --yes --level 3 r
   ```

5. Meson and Ninja
   
   Meson and Ninja are required to build DPDK:
   ```
   sudo apt-get install -y python3-setuptools ninja-build
   wget https://github.com/mesonbuild/meson/releases/download/0.56.0/meson-0.56.0.tar.gz
   tar xvfz meson-0.56.0.tar.gz
   cd meson-0.56.0
   sudo python3 setup.py install
   '''

6. PHC2SYS

   In order to make the generator work, the clock on the ConnectX-6 Dx NIC has to be synchronized with the system clock. For this reason `phc2sys` has to be enabled between the two clocks where the NIC is the master. As an example, to enable it on the network interface `enp181s0f1`:

   ```
   $ sudo ethtool -T enp181s0f1
      Time stamping parameters for enp181s0f1:
      Capabilities:
         hardware-transmit     (SOF_TIMESTAMPING_TX_HARDWARE)
         hardware-receive      (SOF_TIMESTAMPING_RX_HARDWARE)
         hardware-raw-clock    (SOF_TIMESTAMPING_RAW_HARDWARE)
      PTP Hardware Clock: 4
      Hardware Transmit Timestamp Modes:
         off                   (HWTSTAMP_TX_OFF)
         on                    (HWTSTAMP_TX_ON)
      Hardware Receive Filter Modes:
         none                  (HWTSTAMP_FILTER_NONE)
         all                   (HWTSTAMP_FILTER_ALL)
   ```

   The command line is:

   ```
   /usr/sbin/phc2sys -s /dev/ptp4 -c CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256
   ```

## Getting the code 

Clone the main branch of this project with all the submodules

```
git clone --recurse-submodules https://github.com/NVIDIA/5t5g.git
```

## Building

```
cd 5t5g
mkdir -p build && cd build
cmake ..
make -j $(nproc --all)
```


## Running

Receiver and generator should run on two different machines (or the same if the NIC has 2 ports) connected back to back (no switch in the middle). We tested both applications on two servers `GIGABYTE E251-U70` with the following HW/SW components:

HW:
* CPU: Xeon Gold 6240R. 2.4GHz. 24C48T
* Memory: 16 GB DDR4
* BIOS: R06 rev. 5.14 (07/21/2020)
* NIC: ConnectX-6 Dx (MT4125 - MCX623106AE-CDAT)
* GPU: V100-PCIE-32GB
* PCIe external switch between NIC and GPU: `PLX Technology, Inc. PEX 8747 48-Lane, 5-Port PCI Express Gen 3`

SW:
* Operating System: Ubuntu 18.04.5 LTS
* Kernel Version: 5.4.0-53-lowlatency
* GCC: 8.4.0 (Ubuntu 8.4.0-1ubuntu1~18.04)
* Mellanox NIC firmware version: 22.31.1014
* Mellanox OFED driver version: MLNX_OFED_LINUX-5.4-1.0.3.0
* CUDA Version: 11.2
* GPU Driver Version: 455.32.00

Run the receiver:

```
sudo ./receiver/5t5g_receiver -l 0-7 -n 8 -a b5:00.1 -- -g 0
```

Run the generator for 1000 slots with receiver connected to the `0c:42:a1:d1:d0:a1` MAC address:

```
sudo ./generator/5t5g_generator -l 0-9 -n 8 -a b5:00.1,txq_inline_max=0,tx_pp=500 -- -d 0c:42:a1:d1:d0:a1 -i 1000
```

Please note that `-i 0` makes the generator running forever.

## References

For more info about how we use these features in our NVIDIA 5G L1 software implementation, please refer to the GTC'21 session `S32078: vRAN Signal Processing Orchestration over GPU`, E. Agostini.

To have more insights about how to leverage your DPDK network application with GPUDirect RDMA please refer to the [l2fwd-nv](https://github.com/NVIDIA/l2fwd-nv) project.
