#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>




int main(){
    struct sockaddr_in server,client;
    socklen_t serlen,clilen;
    int new,sock;
    sock = socket(AF_INET, SOCK_STREAM, 0);

    server.sin_family=AF_INET;
    server.sin_port=htons(8080);
    server.sin_addr.s_addr=INADDR_ANY;


    serlen=sizeof(server);
    clilen=sizeof(client);

    bind( sock , ( struct sockaddr *) &server , sizeof(server) );
        printf("the server is wokrin \n");
    listen( sock , 1 );
    printf("the server is wokrin \n");

    //new = connect( sock, ( struct sockaddr *) &client, clilen);
    new = accept(sock, ( struct sockaddr *) &client, &clilen);
    printf("the server is wokrin \n");

    char buffer[1000];
    int n = recv( new, buffer, sizeof(buffer) -1 , 0 );
    printf("the server is wokrin \n");

    buffer[n]='\0';

    printf("server = %s",buffer);
    FILE *fp = fopen( buffer , "r" );
    printf("server = %s",buffer);

    char resul[1000];
    if(fp==NULL){
        sprintf(resul,"no file nigga %d",getpid());
        send(new, resul , strlen(resul) ,0);
    printf("server = %s",resul);


    }else{
        sprintf(resul,"pid %d",getpid());
        send(new, resul, strlen(resul),0);
        while(fgets(resul,sizeof(resul),fp)!=NULL){
        send(new,resul, strlen(resul) ,0);
        }
    }
    printf("server = %s",resul);
close(new);
close(sock);
}