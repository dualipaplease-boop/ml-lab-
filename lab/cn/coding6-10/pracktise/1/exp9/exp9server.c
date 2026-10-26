#include <sys/socket.h>
     #include <unistd.h>
     #include <arpa/inet.h>
     #include <stdio.h>
     #include <string.h>


int main(){
    int sockfd,client;
    struct sockaddr_in addr ,addrclient;
    socklen_t addrlen= sizeof(addr);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);


    addr.sin_family=AF_INET;
    addr.sin_port=htons(8000);
    addr.sin_addr.s_addr=INADDR_ANY;


    bind(sockfd, (struct sockaddr *)&addr, addrlen);
    listen(sockfd,1);

    printf("the server is working ...\n");
    socklen_t len= sizeof(addrclient);
    client = accept(sockfd, (struct sockaddr *)&addrclient,&len);
    
    char buff[10000];
    int n = recv(client, buff, sizeof(buff)+1,0);
    buff[n]='\0';
    printf("\nthe file name %s",buff);
    FILE *pf = fopen(buff,"r");
    char temp[10000];
    if (pf==0){
        sprintf(temp,"the file not found . the pid = %d" , getpid());
        send(client, temp, sizeof(temp)+1, 0);
    }else{
        sprintf(temp,"the pid = %d" , getpid());
        send(client, temp, sizeof(temp)+1, 0);
        while(1){
        if(fgets(temp,sizeof(temp), pf)){
        send(client, temp, sizeof(temp)+1, 0);
        }else{break;}

    }
    }

close(client);
    close(sockfd);
    fclose(pf);
    return 0;
}