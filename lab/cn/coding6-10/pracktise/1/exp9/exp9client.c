#include <sys/socket.h>
#include <stdio.h>
     #include <unistd.h>
     #include <stdio.h>
     #include <arpa/inet.h>


int main(){
    int sockfd;
    struct sockaddr_in addrs;
    socklen_t addrlen = sizeof(addrs);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);


    addrs.sin_family=AF_INET;
    addrs.sin_port=htons(8000);
    addrs.sin_addr.s_addr=inet_addr("127.0.0.1");

    connect(sockfd, (struct sockaddr * )&addrs,addrlen);

    printf("the client is working ...\n");

char buff[10000];
    sprintf(buff,"dua.txt");
    printf("buff=%s",buff);
    send(sockfd, buff, sizeof(buff)+1, 0);
    char temp[10000];
    while (1){
        int n = recv(sockfd, temp, sizeof(temp)+1,0);

        if (n<=0){break;}

        printf("\n %s",temp);
    }

   close(sockfd);
    return 0;

}