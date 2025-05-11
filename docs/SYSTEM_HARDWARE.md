# System Hardware Specifications

This document provides detailed information about the hardware configuration of the GH200 Trading System.

## GPU Information

| Property | Value |
|----------|-------|
| **GPU Model** | NVIDIA GH200 480GB |
| **Architecture** | Hopper |
| **CUDA Version** | 12.8 |
| **Driver Version** | 570.124.06 |
| **GPU Memory** | 97871 MiB (~95.6 GB) |
| **Temperature** | 32°C |
| **Power Draw** | 71.15 W |
| **Power Limit** | 900.00 W |

## CPU Information

| Property | Value |
|----------|-------|
| **Architecture** | aarch64 (ARM 64-bit) |
| **CPU Model** | Neoverse-V2 |
| **Cores** | 64 |
| **Threads per Core** | 1 |
| **Sockets** | 1 |
| **NUMA Nodes** | 9 |

### CPU Features
- fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti

## Memory Information

| Property | Value |
|----------|-------|
| **Total RAM** | 525 GB |
| **Used RAM** | 12 GB |
| **Free RAM** | 506 GB |
| **Available RAM** | 490 GB |
| **Swap** | None configured |

## Storage Information

| Mount Point | Filesystem | Size | Used | Available | Use% |
|-------------|------------|------|------|-----------|------|
| / | /dev/vda1 | 3.9 TB | 26 GB | 3.9 TB | 1% |
| /home/ubuntu/inavvi2 | 10.12.69.21:/9b751011-2826-4e31-b0f1-2c1e3c54470c | 6.5 PB | 0 | 6.5 PB | 0% |
| /boot/efi | /dev/vda15 | 98 MB | 6.3 MB | 92 MB | 7% |

## Network Information

| Property | Value |
|----------|-------|
| **Interface** | enp230s0 |
| **IP Address** | 172.26.132.74/22 |
| **MAC Address** | 06:66:a5:2c:7e:5f |
| **Speed** | 100000 Mb/s (100 Gbps) |
| **Duplex** | Full |
| **MTU** | 8942 |
| **Port Type** | FIBRE |
| **Link Status** | Connected |

## System Information

| Property | Value |
|----------|-------|
| **Hostname** | 192-222-50-45 |
| **Kernel** | 6.8.0-1013-nvidia-64k |
| **OS** | Linux (GNU/Linux) |
| **Build** | #14lambdaguest1 SMP PREEMPT_DYNAMIC Sat Sep 14 00:46:47 UTC 2024 |

## Security Mitigations

| Vulnerability | Status |
|---------------|--------|
| Spectre v1 | Mitigation; __user pointer sanitization |
| Spectre v2 | Not affected |
| Meltdown | Not affected |
| L1tf | Not affected |
| MDS | Not affected |
| Spec store bypass | Mitigation; Speculative Store Bypass disabled via prctl |

## Summary

This system features a high-performance NVIDIA GH200 480GB GPU with 95.6 GB of memory, paired with a 64-core ARM Neoverse-V2 processor and 525 GB of RAM. The system has 3.9 TB of local storage and access to a massive 6.5 PB network storage. Network connectivity is provided by a 100 Gbps fiber connection, making this an extremely powerful platform for high-performance computing and AI workloads, particularly suited for trading applications requiring low latency and high throughput.