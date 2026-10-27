#include <sys/socket.h>
     #include <netinet/in.h>
     #include <stdio.h>
     #include <string.h>
          #include <sys/socket.h>
     #include <netinet/in.h>
     #include <arpa/inet.h>


int main(){
    struct sockaddr_in server,client;
    socklen_t serlen,clilen;
    int sock;
    sock = socket(AF_INET, SOCK_DGRAM, 0);

    server.sin_family=AF_INET;
    server.sin_port=htons(8008);
    server.sin_addr.s_addr=inet_addr("127.0.0.1");

       serlen=sizeof(server);
    clilen=sizeof(client);
    printf("the cleient is wokrin \n");
    char resul[1000];
    sprintf(resul,"%s","hello idc hello idc");
    printf("client=%s",resul);

    sendto(sock,resul, strlen(resul),0, (struct sockaddr *)&server,serlen);
    char buffer[1000];
    recvfrom(sock, buffer, sizeof(buffer),0, (struct sockaddr *)&server,&serlen);

    printf("\ncleint =%s",buffer);
}