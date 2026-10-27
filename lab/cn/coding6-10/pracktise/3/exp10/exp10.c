#include <linux/udp.h>
 #include <linux/tcp.h>
 #include <linux/ip.h>
// #include <linux/if_addr.h>
 #include <linux/if_packet.h>
 #include <net/ethernet.h>
 #include <net/if_packet.h>
// #include <netinet/ether.h>
 //#include <netinet/if_ether.h>
 
 #include <linux/if_ether.h>
 #include <linux/icmp.h>
 
 
 /* 
 #include <
 #include <
 #include <
 #include <
 #include <
 #include <
 #include <
 #include <
 #include <
 #include <
*/
     #include <sys/socket.h>
     #include <netinet/in.h>
     #include <arpa/inet.h>
     #include <stdio.h>

int main(){

    int sock;
    struct sockaddr_ll server;
    socklen_t lenser=sizeof(server);
    
    sock=socket(AF_PACKET,SOCK_RAW,htons(ETH_P_ALL));
char buffer[1000];
    recvfrom (sock,buffer,sizeof(buffer),0,(struct sockaddr *)&server,&lenser);


    struct ether_header *e =(struct ether_header * )buffer;

    printf("%02x,%02x,%04x",e->ether_dhost[0],e->ether_shost[0],e->ether_type);

struct iphdr *ip = (struct iphdr *)(buffer+sizeof(struct ether_header ));
printf("%s,%s",inet_ntoa(*(struct in_addr *)&ip->saddr),inet_ntoa(*(struct in_addr *)&ip->daddr));

if(ip->protocol==IPPROTO_TCP){
struct tcphdr *t=(struct tcphdr *)(buffer+sizeof(struct ether_header ) + ip->ihl * 4);
printf("%u,%u",ntohs(t->source),ntohs(t->dest));

}else if(ip->protocol==IPPROTO_ICMP){
struct icmphdr *i=(struct icmphdr *)(buffer+sizeof(struct ether_header ) + ip->ihl * 4);
printf("%u,%u",i->code,i->type);

}else if(ip->protocol==IPPROTO_UDP){
struct udphdr *u=(struct udphdr *)(buffer+sizeof(struct ether_header ) + ip->ihl * 4);
printf("%u,%u",ntohs(u->source),ntohs(u->dest));

}else{
    printf("nothing");
}

}