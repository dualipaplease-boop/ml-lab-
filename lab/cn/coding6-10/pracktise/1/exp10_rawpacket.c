#include <sys/socket.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h> /* the L2 protocols */
#include <sys/types.h>
#include <stdint.h>
#include <linux/if_ether.h>
#include <sys/socket.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>




struct ether_header
{
  uint8_t  ether_dhost[ETH_ALEN];	/* destination eth addr	*/
  uint8_t  ether_shost[ETH_ALEN];	/* source ether addr	*/
  uint16_t ether_type;		        /* packet type ID field	*/
} __attribute__ ((__packed__));


int main(){
    unsigned char buffer[65536];
    int packet_socket ,packet_len;
    struct sockaddr_ll addr;
    socklen_t addr_len = sizeof(addr);


    packet_socket = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));

    packet_len = recvfrom(packet_socket, buffer, sizeof(buffer),
         0, (struct sockaddr *)&addr,
         &addr_len);

 printf("%d\n",packet_len);    
 printf("buffer = %s", buffer);
 struct ether_header *ehter = (struct ether_header *)buffer;
printf("Destination MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
    ehter->ether_dhost[0], ehter->ether_dhost[1],
    ehter->ether_dhost[2], ehter->ether_dhost[3],
    ehter->ether_dhost[4], ehter->ether_dhost[5]);

printf("Source MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
    ehter->ether_shost[0], ehter->ether_shost[1],
    ehter->ether_shost[2], ehter->ether_shost[3],
    ehter->ether_shost[4], ehter->ether_shost[5]);
 printf("ethertyper= 0x%04x",ntohs(ehter->ether_type));

struct iphdr *ip = (struct iphdr*)(buffer+sizeof(struct ether_header));

printf("ip source =%s", inet_ntoa(*(struct in_addr *)&ip->saddr));
printf("ip destination =%s", inet_ntoa(*(struct in_addr *)&ip->daddr));
printf("\nip protocol =%u",ip->protocol);

unsigned char *data = buffer + sizeof(struct ether_header) + ip->ihl * 4;


if (ip->protocol==IPPROTO_UDP){
    printf("\nits udp");
    struct udphdr *udp = (struct udphdr *)data;
 printf("source = %d\n dest= %d", ntohs(udp->source),ntohs(udp->dest));
}else if (ip->protocol==IPPROTO_TCP){
    printf("\nits tcp");
    struct tcphdr *tcp = (struct tcphdr *)data;
  printf("source = %d\n dest= %d", ntohs(tcp->source),ntohs(tcp->dest));

    
}else if (ip->protocol==IPPROTO_ICMP){
    printf("\nits icmp");
 struct icmphdr *icm = (struct icmphdr *) data;
 printf("type = %u\ncode= %u", icm->type,icm->code);
    
}else{
    printf("others");
}

    return 0;
}