
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080

int main()
{
    int server_fd, client_fd;
    struct sockaddr_in server, client;
    socklen_t len = sizeof(client);
    char filename[100];
    char buffer[1024];
    FILE *fp;

    // Create TCP socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);

    // Configure server address
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(PORT);

    // Bind and listen
    bind(server_fd, (struct sockaddr *)&server, sizeof(server));
    listen(server_fd, 1);

    printf("File Server Running...\n");

    // Accept client
    client_fd = accept(server_fd, (struct sockaddr *)&client, &len);

    // Receive filename
    int n = recv(client_fd, filename, sizeof(filename) - 1, 0);
    filename[n] = '\0';

    printf("Requested File: %s\n", filename);

    // Open file
    fp = fopen(filename, "r");

    if (fp == NULL)
    {
        sprintf(buffer, "Server PID: %d\nFile not found.\n", getpid());
        send(client_fd, buffer, strlen(buffer), 0);
    }
    else
    {
        sprintf(buffer, "Server PID: %d\n\n", getpid());
        send(client_fd, buffer, strlen(buffer), 0);

        // Send file contents
        while (fgets(buffer, sizeof(buffer), fp) != NULL)
            send(client_fd, buffer, strlen(buffer), 0);

        fclose(fp);
    }

    close(client_fd);
    close(server_fd);
    return 0;
}
