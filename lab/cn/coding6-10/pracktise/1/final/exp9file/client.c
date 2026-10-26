
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080

int main()
{
    int sockfd;
    struct sockaddr_in server;
    char filename[100];
    char buffer[1024];

    // Create TCP socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    // Configure server address
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    server.sin_addr.s_addr = inet_addr("127.0.0.1");

    // Connect to server
    connect(sockfd, (struct sockaddr *)&server, sizeof(server));

    // Enter filename
    printf("Enter filename: ");
    scanf("%99s", filename);

    // Send filename
    send(sockfd, filename, strlen(filename) + 1, 0);

    printf("\n----- Server Response -----\n");

    // Receive response
    while (1)
    {
        int n = recv(sockfd, buffer, sizeof(buffer) - 1, 0);

        if (n <= 0)
            break;

        buffer[n] = '\0';
        printf("%s", buffer);
    }

    close(sockfd);
    return 0;
}
