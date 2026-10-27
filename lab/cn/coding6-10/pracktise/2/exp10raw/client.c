#include <stdio.h>
#include <sys/socket.h>
#include <net/ethernet.h>//
//#include <net/if_packet.h>//
 #include <arpa/inet.h>

     #include <arpa/inet.h>




//#include <netinet/ether.h>
//#include <netinet/if_ether.h>
//#include <netinet/ether.h>





#include <linux/icmp.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <netinet/in.h>
#include <linux/ip.h>
//#include <linux/if_addr.h>
#include <linux/if_ether.h>//
#include <linux/if_packet.h>


int main(){
int sock;
char buffer[10000];
 struct sockaddr_ll add; //in ifpacket.h
    sock = socket(AF_PACKET,SOCK_RAW, htons(ETH_P_ALL));
    socklen_t lenadd=sizeof(add);
     while(1){
            printf("\n\nstart\n");
    recvfrom(sock,buffer,sizeof(buffer),0,(struct sockaddr *)&add,&lenadd);
    struct ether_header *eth=(struct ether_header *)&buffer;
    printf("%02x,%02x,%02x,%02x,%02x,%02x,\n",eth->ether_dhost[0],eth->ether_dhost[1],eth->ether_dhost[2],eth->ether_dhost[3],eth->ether_dhost[4],eth->ether_dhost[5]);
    printf("%02x,%02x,%02x,%02x,%02x,%02x,\n",eth->ether_shost[0],eth->ether_shost[1],eth->ether_shost[2],eth->ether_shost[3],eth->ether_shost[4],eth->ether_shost[5]);
    printf("%04x\n",eth->ether_type);

    struct iphdr *iph = (struct iphdr *)  (buffer + sizeof(struct ether_header));
    printf("%s,%s\n",inet_ntoa( *(struct in_addr * ) &iph->saddr ),inet_ntoa(*(struct in_addr *) &iph->daddr));
    if(iph->protocol==IPPROTO_TCP){
        printf("its tcp");
        struct tcphdr *tcph= (struct tcphdr *)(buffer + sizeof(struct ether_header) + (iph->ihl * 4));
        printf("%u,%u",ntohs(tcph->source),ntohs(tcph->dest));
    }else if(iph->protocol==IPPROTO_UDP){
    printf("its udp");

    struct udphdr *udph = (struct udphdr *)(buffer+sizeof(struct ether_header) + (iph->ihl * 4));
    printf("%u,%u",ntohs(udph->source),ntohs(udph->dest));
    }else if(iph->protocol==IPPROTO_ICMP){
        printf("its icmp");
        struct icmphdr *icmph =(struct icmphdr *)(buffer+sizeof(struct ether_header) + (iph->ihl * 4));
        printf("%u,%u",icmph->type,icmph->code);
}

        printf("\n\nend\n");
}



}