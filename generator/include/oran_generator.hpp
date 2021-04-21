// Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef ORAN_HPP_
#define ORAN_HPP_

#include <inttypes.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Bits manipulation
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Overcome NVCC error: 
 * "Bitfields and field types containing bitfields are not supported in packed structures and unions for device compilation!"
 * Inspired by https://github.com/preshing/cpp11-on-multicore/blob/master/common/bitfield.h
 * Only supports 8, 16, 32 bit sized bitfields
 */
template <typename T, int Offset, int Bits>
class __attribute__((__packed__)) Bitfield {
    static_assert(Offset + Bits <= (int)sizeof(T) * 8, "Member exceeds bitfield boundaries");
    static_assert(Bits < (int)sizeof(T) * 8, "Can't fill entire bitfield with one member");
    static_assert(sizeof(T) == sizeof(uint8_t) ||
                      sizeof(T) == sizeof(uint16_t) ||
                      sizeof(T) == sizeof(uint32_t),
                  "Size not supported by bitfield");
    static const T Maximum = (T(1) << Bits) - 1;
    static const T Mask    = Maximum << Offset;

    T field;
    T be_to_le(T value)
    {
        T tmp = value;
        if(sizeof(T) == sizeof(uint16_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        }
        else if(sizeof(T) == sizeof(uint32_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF000000) >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        }
        return tmp;
    }
    T le_to_be(T value)
    {
        T tmp = value;
        if(sizeof(T) == sizeof(uint16_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF00) >> 8;
            tmp |= (value & 0x00FF) << 8;
        }
        else if(sizeof(T) == sizeof(uint32_t))
        {
            tmp = 0;
            tmp |= (value & 0xFF000000) >> 24;
            tmp |= (value & 0x00FF0000) >> 8;
            tmp |= (value & 0x0000FF00) << 8;
            tmp |= (value & 0x000000FF) << 24;
        }
        return tmp;
    }

public:
    void operator=(T value)
    {
        value &= Maximum;
        // v must fit inside the bitfield member
        assert(value <= Maximum);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = be_to_le(field);
#endif
        field = (field & ~Mask) | (value << Offset);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        field = le_to_be(field);
#endif
    }

    operator T()
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (T)(be_to_le(field) >> Offset) & Maximum;
#else
        return (T)(field >> Offset) & Maximum;
#endif
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Ethernet generic
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define ORAN_ETHER_ADDR_LEN 6

struct oran_ether_addr
{
    uint8_t addr_bytes[ORAN_ETHER_ADDR_LEN]; /**< Addr bytes in tx order */
} __attribute__((__aligned__(2)));

struct oran_ether_hdr
{
    struct oran_ether_addr d_addr;     /**< Destination address. */
    struct oran_ether_addr s_addr;     /**< Source address. */
    uint16_t               ether_type; /**< Frame type. */
} __attribute__((__aligned__(2)));

struct oran_vlan_hdr
{
    uint16_t vlan_tci;  /**< Priority (3) + CFI (1) + Identifier Code (12) */
    uint16_t eth_proto; /**< Ethernet type of encapsulated frame. */
} __attribute__((__packed__));

struct oran_eth_hdr
{
    struct oran_ether_hdr eth_hdr;
    struct oran_vlan_hdr  vlan_hdr;
};

#define ORAN_ETH_HDR_SIZE (         \
    sizeof(struct oran_ether_hdr) + \
    sizeof(struct oran_vlan_hdr))


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// eCPRI generic
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* O-RAN specs v01.00
*/
#define ETHER_TYPE_ECPRI 0xAEFE
#define ORAN_DEF_ECPRI_VERSION 1
#define ORAN_DEF_ECPRI_RESERVED 0
//Forcing one eCPRI msg x Ethernet frame
#define ORAN_ECPRI_CONCATENATION_NO 0
#define ORAN_ECPRI_CONCATENATION_YES 1

#define ORAN_ECPRI_HDR_OFFSET ORAN_ETH_HDR_SIZE

#define ORAN_MAX_SUBFRAME_ID 10
#define ORAN_MAX_SLOT_ID 2 //Assuming TTI == 500

/* Section 3.1.3.1.4 */
#define ECPRI_MSG_TYPE_IQ 0x0
#define ECPRI_MSG_TYPE_RTC 0x2
#define ECPRI_MSG_TYPE_ND 0x5

/* eCPRI transport header as defined in ORAN-WG4.CUS.0-v01.00 3.1.3.1 */
struct oran_ecpri_hdr
{
    /*
    LITTLE ENDIAN FORMAT (8 bits):
    -----------------------------------------------------
    | ecpriVersion | ecpriReserved | ecpriConcatenation |
    -----------------------------------------------------
    |       4      |       3       |        1           |
    -----------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> ecpriVersion;
        Bitfield<uint8_t, 4, 3> ecpriReserved;
        Bitfield<uint8_t, 7, 1> ecpriConcatenation;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> ecpriConcatenation;
        Bitfield<uint8_t, 1, 3> ecpriReserved;
        Bitfield<uint8_t, 4, 4> ecpriVersion;
    };
#endif

    uint8_t  ecpriMessage;
    uint16_t ecpriPayload;
    union
    {
        uint16_t ecpriRtcid;
        uint16_t ecpriPcid;
    };
    uint8_t ecpriSeqid;

    /*
    BIG ENDIAN FORMAT (8 bits):
    -----------------------------
    | ecpriEbit | ecpriSubSeqid |
    -----------------------------
    |     1     |       7       |
    -----------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> ecpriEbit;
        Bitfield<uint8_t, 1, 7> ecpriSubSeqid;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 7> ecpriSubSeqid;
        Bitfield<uint8_t, 7, 1> ecpriEbit;
    };
#endif

} __attribute__((__packed__));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Message specific O-RAN header
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum oran_pkt_dir
{
    DIRECTION_UPLINK = 0,
    DIRECTION_DOWNLINK
};

/* Section 5.4.4.2 */
#define ORAN_DEF_PAYLOAD_VERSION 1
/* Section 5.4.4.3 */
#define ORAN_DEF_FILTER_INDEX 0
/* Section 5.4.5.2 */
#define ORAN_RB_ALL 0
#define ORAN_RB_OTHER_ALL 1
/* Section 5.4.5.3 */
#define ORAN_SYMCINC_NO 0
#define ORAN_SYMCINC_YES 1
/* Section 5.4.5.5 */
#define ORAN_REMASK_ALL 0x0FFFU
/* Section 5.4.5.7 */
#define ORAN_ALL_SYMBOLS 14U
/* Section 5.4.5.8 */
#define ORAN_EF_NO 0
#define ORAN_EF_YES 1
/* Section 5.4.5.9 */
#define ORAN_BEAMFORMING_NO 0x0000

#define ORAN_MAX_PRB_X_SECTION 255
#define ORAN_MAX_PRB_X_SLOT 273
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// U-plane
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define SLOT_NUM_SYMS 14U                                      /* number of symbols in a slot */
#define PRB_NUM_RE 12U                                         /* number of REs in a PRB */
#define UD_IQ_WIDH_MAX 16                                      /* maximum allowed IQ bit width */
#define PRB_SIZE(iq_width) ((iq_width * 2U * PRB_NUM_RE) / 8U) /* in bytes */
#define PRB_SIZE_16F PRB_SIZE(16)

/* Not considering section id for data placement yet */
#define ORAN_DEF_SECTION_ID 0

#define ORAN_DEF_NO_COMPRESSION 0
/* header of the IQ data frame U-Plane message in O-RAN FH, all the way up to
* and including symbolid (the fuchsia part of Table 6-2 in the spec) */
struct oran_umsg_iq_hdr
{
    /*
    BIG ENDIAN FORMAT (8 bits):
    ---------------------------------------------------
    | Data Direction | Payload Version | Filter Index |
    ---------------------------------------------------
    |     1          |          3      |      4       |
    ---------------------------------------------------
*/

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 1> dataDirection;
        Bitfield<uint8_t, 1, 3> payloadVersion;
        Bitfield<uint8_t, 4, 4> filterIndex;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint8_t, 0, 4> filterIndex;
        Bitfield<uint8_t, 4, 3> payloadVersion;
        Bitfield<uint8_t, 7, 1> dataDirection;
    };
#endif

    uint8_t frameId;

/*
    BIG ENDIAN FORMAT (16 bits):
    ----------------------------------
    | subframeId | slotId | symbolId |
    ----------------------------------
    |    4       |    6   |    6     |
    ----------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 4>  subframeId;
        Bitfield<uint16_t, 4, 6>  slotId;
        Bitfield<uint16_t, 10, 6> symbolId;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint16_t, 0, 6>  symbolId;
        Bitfield<uint16_t, 6, 6>  slotId;
        Bitfield<uint16_t, 12, 4> subframeId;
    };
#endif

} __attribute__((__packed__));

/* A struct for the section header of uncompressed IQ U-Plane message.
* No compression is used, so the compression header is omitted.
*/
struct oran_u_section_uncompressed
{
/*
    BIG ENDIAN FORMAT (16 bits):
    ---------------------------------------------
    | sectionId | rb | symInc | unused_startPrbu |
    ---------------------------------------------
    |    12     | 1  |    1   |         2        |
    ---------------------------------------------
*/
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 12>  sectionId;
        Bitfield<uint32_t, 12, 1>  rb;
        Bitfield<uint32_t, 13, 1>  symInc;
        Bitfield<uint32_t, 14, 10> startPrbu;
        Bitfield<uint32_t, 24, 8>  numPrbu;
    };
#else
    union __attribute__((__packed__))
    {
        Bitfield<uint32_t, 0, 8>   numPrbu;
        Bitfield<uint32_t, 8, 10>  startPrbu;
        Bitfield<uint32_t, 18, 1>  symInc;
        Bitfield<uint32_t, 19, 1>  rb;
        Bitfield<uint32_t, 20, 12> sectionId;
    };
#endif
    /* NOTE: no compression header */
} __attribute__((__packed__));

/* per-eth frame overhead. NOTE: one eCPRI message per eth frame assumed */
#define ORAN_IQ_HDR_OFFSET ( \
    ORAN_ECPRI_HDR_OFFSET +  \
    sizeof(struct oran_ecpri_hdr))

#define ORAN_IQ_STATIC_OVERHEAD ( \
    ORAN_IQ_HDR_OFFSET +          \
    sizeof(struct oran_umsg_iq_hdr))

/* per-section overhead */
#define ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD ( \
    sizeof(struct oran_u_section_uncompressed))

struct oran_umsg_hdrs
{
    struct oran_eth_hdr                ethvlan;
    struct oran_ecpri_hdr              ecpri;
    struct oran_umsg_iq_hdr            iq_hdr;
    struct oran_u_section_uncompressed sec_hdr;
};

#define ORAN_UMSG_IQ_HDR_SIZE sizeof(struct oran_umsg_hdrs)
#define ORAN_IQ_HDR_SZ (ORAN_IQ_STATIC_OVERHEAD + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define F_TYPE inline

F_TYPE uint8_t oran_umsg_get_frame_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->frameId;
}
F_TYPE uint8_t oran_umsg_get_subframe_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->subframeId;
}
F_TYPE uint8_t oran_umsg_get_slot_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->slotId;
}
F_TYPE uint8_t oran_umsg_get_symbol_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_umsg_iq_hdr*)(mbuf_payload + ORAN_IQ_HDR_OFFSET))->symbolId;
}
//oran_u_section_uncompressed
F_TYPE uint16_t oran_umsg_get_start_prb(uint8_t* mbuf_payload)
{
    return (uint16_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->startPrbu;
}
F_TYPE uint8_t oran_umsg_get_num_prb(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->numPrbu;
}
F_TYPE uint8_t oran_umsg_get_section_id(uint8_t* mbuf_payload)
{
    return (uint8_t)((struct oran_u_section_uncompressed*)(mbuf_payload + ORAN_IQ_STATIC_OVERHEAD))->sectionId;
}

// oran_ecpri_hdr
F_TYPE uint8_t oran_get_sequence_id(uint8_t* mbuf_payload)
{
    return ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriSeqid;
}

F_TYPE uint16_t oran_umsg_get_flowid(uint8_t* mbuf_payload)
{
    return (uint16_t)(((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPcid << 8 | ((struct oran_ecpri_hdr*)(mbuf_payload + ORAN_ECPRI_HDR_OFFSET))->ecpriPcid >> 8);
}

F_TYPE uint32_t oran_get_offset_from_hdr(uint8_t* pkt, int flow_index, int symbols_x_slot, int prbs_per_symbol, int prb_size)
{
    return (flow_index * symbols_x_slot * prbs_per_symbol * prb_size) +
           (oran_umsg_get_symbol_id(pkt) * prbs_per_symbol * prb_size) +
           (oran_umsg_get_start_prb(pkt) * prb_size);
}

#endif //ifndef ORAN_HPP_
