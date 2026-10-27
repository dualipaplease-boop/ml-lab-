#include <netinet/in.h>
     #include <sys/socket.h>
          #include <stdio.h>
          #include <string.h>
               #include <unistd.h>
                    #include <stdio.h>






int main(){
    int sock;
    sock=socket(AF_INET,SOCK_STREAM,0);
    struct sockaddr_in server,client;
    socklen_t lenser=sizeof(server);
    socklen_t lencli=sizeof(client);

    server.sin_family=AF_INET;
    server.sin_port=htons(8080);
    server.sin_addr.s_addr=INADDR_ANY;

    bind(sock,(struct sockaddr *)&server,lenser);
    listen(sock,1);
    int new= accept(sock,(struct sockaddr *)&client,&lencli);
    char buffer[1000],resul[1000];
    recv(new,buffer,sizeof(buffer),0);
    FILE *fp=fopen(buffer,"r");
    if(fp==NULL){
        sprintf(resul,"nigga u got no file%d",getpid());
    send(new,resul,strlen(resul),0);
        
    }else{
        sprintf(resul," file%d",getpid());
    send(new,resul,strlen(resul),0);
    while(1){
    
        if(fgets(resul,sizeof(resul),fp)==NULL){
                break;}        send(new,resul,strlen(resul),0);
        }
    }
}