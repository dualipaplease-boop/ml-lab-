
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

int main()
{
    int sock;
    struct sockaddr_in server, client;

    sock = socket(AF_INET, SOCK_DGRAM, 0);

    server.sin_family = AF_INET;
    server.sin_port = htons(8008);
    server.sin_addr.s_addr = INADDR_ANY;

    bind(sock, (struct sockaddr *)&server, sizeof(server));

    printf("Server is working...\n");

    char buff[1000];
    socklen_t clilen = sizeof(client);

    int n = recvfrom(sock, buff, sizeof(buff) - 1, 0,
                     (struct sockaddr *)&client, &clilen);

    if (n > 0)
    {
        buff[n] = '\0';
        printf("Client message = %s\n", buff);
    }

    char *msg = "hello my dear client";
    sendto(sock, msg, strlen(msg) + 1, 0,
           (struct sockaddr *)&client, clilen);

    close(sock);
    return 0;
}
