#include <sys/socket.h>
     #include <time.h>
      #include <stdio.h>
      #include <string.h>
      #include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>


int main(){
    int sock;
    struct sockaddr_in server, client;
    sock= socket(AF_INET,SOCK_DGRAM, 0);
    
    server.sin_family=AF_INET;
    server.sin_port=htons(8008);
    server.sin_addr.s_addr=INADDR_ANY;
    
    socklen_t serlen=sizeof(server);
    socklen_t clilen=sizeof(client);

    bind(sock, (const struct sockaddr *)&server ,serlen);

printf("servering is working so yeahh ");
char buff[1000];

    recvfrom(sock,buff,sizeof(buff),0,( struct sockaddr *)&client,&clilen);
printf("the cleint message = %s",buff);

    time_t t =time(NULL);

    char msg[100];
    sprintf(msg,"%s",ctime(&t));
    sendto(sock, msg,strlen(msg)+1,0, (const struct sockaddr *)&client,clilen);

    
}