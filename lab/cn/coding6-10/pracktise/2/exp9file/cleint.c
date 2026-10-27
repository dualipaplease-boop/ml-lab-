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
    sock = socket(AF_INET, SOCK_STREAM, 0);

    server.sin_family=AF_INET;
    server.sin_port=htons(8080);
    server.sin_addr.s_addr=inet_addr("127.0.0.1");



    serlen=sizeof(server);
    clilen=sizeof(client);
    printf("the cleient is wokrin \n");

connect(sock, ( struct sockaddr *) &server , serlen);


    char resul[1000];

    sprintf(resul,"%s","dua.txt");
    printf("\nclient=%s",resul);

    send(sock , resul , strlen(resul)+1, 0 );
    char buffer[1000];
    printf("\nclient=%s",resul);

    int n;
    while(1){
     n = recv(sock, buffer, sizeof(buffer), 0);
    printf("\nclient=%s",resul);

     if(n<=0){break;}
          buffer[n]='\0';
    
    
    printf("\ncleint n=%s",buffer);

}

}