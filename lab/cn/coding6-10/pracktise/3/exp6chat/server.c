#include <netinet/in.h>
     #include <sys/socket.h>
          #include <stdio.h>
          #include <string.h>
          #include <time.h>
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
    server.sin_addr.s_addr=INADDR_ANY;
    char buffer[1000],resul[1000]="";


    bind(sock,(struct sockaddr *)&server,lenser);
printf("vdsvafas %s\n",buffer);
    int n = recvfrom( sock , buffer ,sizeof(buffer) , 0, (struct sockaddr *) &client, &lencli);
    buffer[n]='\0';
    printf("vdsvafas%s\n",buffer);
int i=0;
char tmp[100];
    while(sscanf( buffer + i ,"%s" , tmp) ==1){
        if(strcmp(tmp,"idc")==0){
            strcat(resul,"i dont care");
            strcat(resul," ");
        }else{
            strcat(resul,tmp);
            strcat(resul," ");
        }
            
            i = i + strlen(tmp) + 1 ;
    }



printf("%s",resul);
    sendto(sock , resul , strlen(resul)-1, 0 , (struct sockaddr *) &client, lencli);


}