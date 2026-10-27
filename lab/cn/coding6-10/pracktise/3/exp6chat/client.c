#include <netinet/in.h>
     #include <sys/socket.h>
          #include <stdio.h>
          #include <string.h>
               #include <sys/socket.h>
     #include <netinet/in.h>
     #include <arpa/inet.h>
int main(){
    int sock;
    sock=socket(AF_INET,SOCK_DGRAM,0);
    struct sockaddr_in server,client;
    socklen_t lenser=sizeof(server);
    socklen_t lencli=sizeof(client);

    server.sin_family=AF_INET;
    server.sin_port=htons(8089);
    server.sin_addr.s_addr=inet_addr("127.0.0.1");
    char buffer[1000],resul[1000];
    sprintf(resul,"%s","hello idc hello");
printf("adsadad%s",resul);



    sendto(sock , resul , strlen(resul) , 0 , (struct sockaddr *) &server ,lenser );
    int n = recvfrom(sock, buffer, sizeof(buffer)+1, 0 ,( struct sockaddr *) &server, &lenser);


printf("dasdasda \n\n%s",buffer);
return 0;

}