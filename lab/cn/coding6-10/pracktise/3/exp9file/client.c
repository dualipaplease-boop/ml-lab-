#include <netinet/in.h>
     #include <sys/socket.h>
          #include <stdio.h>
          #include <string.h>




int main(){
    int sock;
    sock=socket(AF_INET,SOCK_STREAM,0);
    struct sockaddr_in server,client;
    socklen_t lenser=sizeof(server);
    socklen_t lencli=sizeof(client);

    server.sin_family=AF_INET;
    server.sin_port=htons(8080);
    server.sin_addr.s_addr=inet_addr("127.0.0.1");

    connect(sock,(struct sockaddr *)&server,lenser);
    char buffer[1000],resul[1000];
        sprintf(resul,"%s","dua.txt");

        send(sock,resul,strlen(resul),0);
        int n ;
while(1){
    n=recv(sock,buffer,sizeof(buffer),0);
    if(n<=0){break;}
    buffer[n]='\0';
    printf("%s",buffer);

}

}