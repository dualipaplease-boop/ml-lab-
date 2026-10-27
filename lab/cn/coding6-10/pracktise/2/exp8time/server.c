#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <time.h>



int main(){
    struct sockaddr_in server,client;
    socklen_t serlen,clilen;
    int sock;
    sock = socket(AF_INET, SOCK_DGRAM, 0);

    server.sin_family=AF_INET;
    server.sin_port=htons(8008);
    server.sin_addr.s_addr=INADDR_ANY;


    serlen=sizeof(server);
    clilen=sizeof(client);

    bind(sock, (struct sockaddr *)&server,serlen);

    printf("the server is wokrin \n");
char buffer[1000];
    int n = recvfrom(sock, buffer, sizeof(buffer)+1,0, (struct sockaddr *)&client,&clilen);
    buffer[n]='\0';

    printf("server = %s",buffer);
 time_t t=time(NULL);
    char resul[1000];

    sprintf(resul,"%s",ctime(&t));

    printf("\nserver n = %s",resul);
    sendto(sock,resul, strlen(resul) ,0, (struct sockaddr *) &client , clilen);


}