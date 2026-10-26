
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <net/ethernet.h>
#include <linux/ip.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>

#define SIZE 65536

int main(void)
{
    unsigned char buffer[SIZE];
    int packet_socket, packet_len;
    struct sockaddr_ll addr;
    socklen_t addr_len;

    packet_socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

    if (packet_socket < 0) {
        perror("socket");
        return 1;
    }

    printf("Packet Capture Started (Ctrl+C to stop)\n");

    while (1) {
        addr_len = sizeof(addr);

        packet_len = recvfrom(packet_socket, buffer, sizeof(buffer),
                              0, (struct sockaddr *)&addr, &addr_len);

        if (packet_len < 0) {
            perror("recvfrom");
            break;
        }

        if (packet_len < (int)sizeof(struct ether_header))
            continue;

        struct ether_header *ehter =
            (struct ether_header *)buffer;

        printf("\nPacket Length: %d bytes\n", packet_len);

        printf("Destination MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
               ehter->ether_dhost[0], ehter->ether_dhost[1],
               ehter->ether_dhost[2], ehter->ether_dhost[3],
               ehter->ether_dhost[4], ehter->ether_dhost[5]);

        printf("Source MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
               ehter->ether_shost[0], ehter->ether_shost[1],
               ehter->ether_shost[2], ehter->ether_shost[3],
               ehter->ether_shost[4], ehter->ether_shost[5]);

        unsigned short protocol = ntohs(ehter->ether_type);
        printf("EtherType: 0x%04X\n", protocol);

        if (protocol != ETH_P_IP ||
            packet_len < (int)(sizeof(struct ether_header) +
                               sizeof(struct iphdr)))
            continue;

        struct iphdr *ip =
            (struct iphdr *)(buffer + sizeof(struct ether_header));

        int ip_header_len = ip->ihl * 4;

        if (ip->version != 4 || ip->ihl < 5 ||
            packet_len < (int)sizeof(struct ether_header) + ip_header_len)
            continue;

        printf("IP Source: %s\n",
               inet_ntoa(*(struct in_addr *)&ip->saddr));

        printf("IP Destination: %s\n",
               inet_ntoa(*(struct in_addr *)&ip->daddr));

        printf("IP Protocol: %u\n", ip->protocol);

        unsigned char *data =
            buffer + sizeof(struct ether_header) + ip_header_len;

        int data_len = packet_len -
                       (int)sizeof(struct ether_header) - ip_header_len;

        if (ip->protocol == IPPROTO_TCP) {
            if (data_len < (int)sizeof(struct tcphdr))
                continue;

            struct tcphdr *tcp = (struct tcphdr *)data;
            int tcp_header_len = tcp->doff * 4;

            if (tcp->doff < 5 || data_len < tcp_header_len)
                continue;

            printf("TCP | Source Port: %u | Destination Port: %u\n",
                   ntohs(tcp->source), ntohs(tcp->dest));

        } else if (ip->protocol == IPPROTO_UDP) {
            if (data_len < (int)sizeof(struct udphdr))
                continue;

            struct udphdr *udp = (struct udphdr *)data;

            printf("UDP | Source Port: %u | Destination Port: %u\n",
                   ntohs(udp->source), ntohs(udp->dest));

        } else if (ip->protocol == IPPROTO_ICMP) {
            if (data_len < (int)sizeof(struct icmphdr))
                continue;

            struct icmphdr *icm = (struct icmphdr *)data;

            printf("ICMP | Type: %u | Code: %u\n",
                   icm->type, icm->code);

        } else {
            printf("Other protocol\n");
        }
    }

    close(packet_socket);
    return 0;
}
