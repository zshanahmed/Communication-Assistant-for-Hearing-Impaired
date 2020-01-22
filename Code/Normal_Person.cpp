#include<stdio.h>
#include<winsock2.h> 
#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include <assert.h>

#pragma comment(lib,"ws2_32.lib") //Winsock Library

using namespace cv;
using namespace std;

int main(int argc , char *argv[])
{
    WSADATA wsa;

	printf("-------------------------------------------------------------------------------\n");
	printf("			  Establishing Connection			\n");
	printf("-------------------------------------------------------------------------------\n");
    printf("Initialising Winsock...\n");

	if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
    {
        printf("Failed. Error Code : %d\n",WSAGetLastError());
        return 1;
    }

    printf("Initialised\n");
	
	SOCKET s;
	s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	
	if(s==INVALID_SOCKET)
	{
		printf("Could not create socket: %d\n", WSAGetLastError());
	}

	printf("Socket Created.\n");

	struct sockaddr_in server;
	
	server.sin_addr.s_addr = inet_addr("127.0.0.1");
    server.sin_family = AF_INET;
    server.sin_port = htons(8888);

	if( bind(s ,(struct sockaddr *)&server , sizeof(server)) == SOCKET_ERROR)
    {
        printf("Bind failed with error code : %d\n" , WSAGetLastError());
    }
     
    printf("Bind done\n");
	
	//Listen to incoming connections
    listen(s , 3);
     
    //Accept and incoming connection
    puts("Waiting for incoming connections...");
    
	int c;
	SOCKET new_socket;
	struct sockaddr_in client;

    c = sizeof(struct sockaddr_in);
    
	//Initializing for the Text Send to Dumb person
	
	int total_Input_Words=0;
    int total_Vocab_Words=8;
    string input_line,str;
    string input_words[100];
    string vocab_words[8] = {"YES","WAALAIKUM-US-SALAM","HOW-ARE-YOU","HOW-LONG","MEDICINE","TAKE","PRESCRIPTION","."};
				
	//First of all we will have a receive flag. When it is 1 our socket will be in receive mode
	//else it will be in sending mode
	int Recv_Flag;

	//Firstly set Recv_Flag to 1 because we have assumed that it will receive first
	Recv_Flag=1;

	while((new_socket = accept(s , (struct sockaddr *)&client, &c))!=INVALID_SOCKET)
	{
		printf("\n-------------------------------------------------------------------------------\n");
		printf("			  Connection Established			\n");
		printf("-------------------------------------------------------------------------------\n");

		while(1)
		{
			if(Recv_Flag==1)
			{
				printf("\n\n-------------------------------------------------------------------------------\n");
				printf("			Normal Person Receiving Mode			\n");
				printf("-------------------------------------------------------------------------------\n");

				const int buffSize=1000;
				char server_reply[buffSize];
				int recv_size;
				printf("\nReceived Text: ");
				
				while((recv_size = recv(new_socket , server_reply , buffSize , 0)) != SOCKET_ERROR)
				{
					server_reply[recv_size] = '\0';
					printf(server_reply);
					printf(" ");
					string reply=server_reply;
					if(strcmp(reply.c_str(),".")==0)
					{
						Recv_Flag=0;
						break;
					}
				}
			}

			if(Recv_Flag==0)
			{
				input_line = "";
				total_Input_Words=0;
				printf("\n\n-------------------------------------------------------------------------------\n");
				printf("			Normal Person Sending Mode			\n");
				printf("-------------------------------------------------------------------------------\n");
				cout<<"\nPlease Enter Sentence (Terminated by a '.') :\n"<<endl;
				getline(cin,input_line);
 
			    istringstream iss(input_line);
			    while (getline(iss, str, ' ' ))
				{
 					input_words[total_Input_Words] = str.c_str();
					total_Input_Words++;
				}
				printf("\nWords Selected From Vocabulary Matching:\n");
			    for(int i=0;i<total_Input_Words;i++)
				{
					for(int j=0;j<total_Vocab_Words;j++) 
					{
						if (input_words[i]==vocab_words[j])
						{
							cout<<"\n"<<input_words[i].c_str();
							if(send(new_socket , input_words[i].c_str(), strlen(input_words[i].c_str()) , 0) < 0)
							{
								puts("Send failed");
								return 1;
							}
							Sleep(3000);
							break;
						}
					}
				}
				Recv_Flag=1;
			}
		}
	}

	closesocket(s);
	WSACleanup();
    
	return 0;
}
