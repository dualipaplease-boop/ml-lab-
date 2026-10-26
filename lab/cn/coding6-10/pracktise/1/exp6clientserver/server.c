#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

int main(){
    int sock;
    struct sockaddr_in server, client;
    sock = socket(AF_INET,SOCK_DGRAM, 0);
    
    server.sin_family=AF_INET;
    server.sin_port=htons(8008);
    server.sin_addr.s_addr=INADDR_ANY;
    
    socklen_t clilen=sizeof(client);

    printf("hello\n");

    bind(sock, ( struct sockaddr *)&server ,sizeof(server));
   printf("hello");

    printf("servering is working so yeahh \n");

   printf("hello");

char buff[1000];
printf("hello\n");


    int n = recvfrom(sock,buff,sizeof(buff)-1,0, (struct sockaddr *)&client , &clilen);


    buff[n]='\0';
    printf("the cleint message = %s",buff);

    char *msg="hello mydear client";
    sendto(sock, msg,strlen(msg)+1,0, ( struct sockaddr *)&client , clilen);
}