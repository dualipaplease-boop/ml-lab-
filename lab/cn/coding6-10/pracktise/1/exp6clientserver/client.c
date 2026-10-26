#include <sys/socket.h>
 #include <sys/socket.h>
     #include <netinet/in.h>
     #include <arpa/inet.h>
     #include <stdio.h>
     #include <string.h>

int main(){
    int sock;
    struct sockaddr_in server, client;
    sock= socket(AF_INET,SOCK_DGRAM, 0);
    
    server.sin_family=AF_INET;
    server.sin_port=htons(8008);
    server.sin_addr.s_addr=inet_addr("127.0.0.1");
    
    socklen_t serlen=sizeof(server);


printf("cleinet  is working so yeahh ");

char *msg="hello mydear server";
    sendto(sock, msg,strlen(msg)+1,0, (const struct sockaddr *)&server,serlen);

char buff[1000];
    recvfrom(sock,buff,sizeof(buff)-1,0,( struct sockaddr *)&server,&serlen);
printf("the server message = %s",buff);
    
}