#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#pragma comment(lib,"ws2_32.lib") //Winsock Library

#include <stdio.h>
#include <winsock2.h>
#include <Windows.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <queue>
#include "opencv2/highgui/highgui.hpp"
#include "iostream"
#include "cmath"
#include <assert.h>
 
using namespace cv;
using namespace std;
const int TABLE_SIZE = 12;

//To use SURF Featurer Detector add the library #include "opencv2/nonfree/features2d.hpp"
//and also add additional depndency opencv_nonfree2410d.lib and opencv_flann2410d for FLANN Matcher

//In this Program will predict our frame from the camera on run time. Then we will save this frame in folder
//as a temporary image. Then we will read this temp image and predict on run time. Then we will read the next frame
//without any time delay, save it, load it and then predict. When we will quit our program the image will be removed.

//Calculating the element occuring most frequently in an array
float freqElementInArray(float response[],int elements)
{
	int i,j;
	int popular = response[0];
	int temp=0, tempCount, count=1;
	for (i=0;i<elements;i++)
    {
        tempCount = 0;
        temp=response[i];
        tempCount++;
            for(j=i+1;j<elements;j++)
        {
            if(response[j] == temp)
            {
                tempCount++;
                if(tempCount > count)
                {
                    popular = temp;
                    count = tempCount;
                }
            }
        }
    }
	return popular;
}

///Program to resize the Image

void resizeImage(string winname, Mat &image )
{
	int rows=image.rows;
	int cols=image.cols;
	int rows1=rows,cols1=cols;
	if(rows>500)
	{
		rows1=500;
		cols1=500*cols/rows;
	}
	resizeWindow(winname,cols1,rows1);
}

//Initializing the Hash Table Definitions

class MapADT {
public:
  MapADT();
  MapADT(string filename);

  const string& find(string key);  
  void insert(string key, const string& value);

  void print();
protected:
  unsigned int hash(string key);

  int find_index(string key, bool override_duplicated_key);

  const static unsigned int size_max = 100;
  string keys[size_max];
  string values[size_max];
};

MapADT::MapADT() {
  for (int i = 0; i < size_max; i++) {
    keys[i] = string();
    values[i] = string();
  }
}

unsigned int MapADT::hash(string k)
{
  unsigned int value = 0 ;
  for (int i = 0; i < k.length(); i++)
    value = 37*value + k[i];
  return value;
}

void MapADT::print() {
  cout << "{";
  for (int i = 0; i < size_max; i++)
    if (!keys[i].empty())
      cout << keys[i] << ":" << values[i] << ", ";
  cout << "}" << endl;
}

int MapADT::find_index(string key, bool override_duplicate_key = true) {     
  unsigned int h = hash(key) % size_max, offset = 0, index;
  
  while (offset < size_max) {
    index = (h + offset) % size_max;

    // empty index for new entry with key `key`
    // or find the index of key `key` in hash table
    // if `override_duplicate_key` is `false`, return a new, unused index, used in DictionaryADT
    if (keys[index].empty() ||
        (override_duplicate_key && keys[index] == key))
      return index;

    offset++;
  }
  return -1;
}

void MapADT::insert(string key, const string& value) {
  int index = find_index(key);
  if (index == -1) {
    cerr << "Table is full!" << endl;
    return;
  }

  keys[index] = key;
  values[index] = value;
}

const string& MapADT::find(string key) {
  int index = find_index(key);
  if (index != -1)
    return values[index];

  return "";
}


class DictionaryADT : public MapADT {
public:
  DictionaryADT() : MapADT() {}
  DictionaryADT(string filename);

  void insert(const string& key, const string& value);
  queue<string> find_all(const string& key);
  queue<string> remove(const string& key);
};

void DictionaryADT::insert(const string& key, const string& value) {
  int index = find_index(key, false);
  if (index == -1) {
    cerr << "Table is full!" << endl;
    return;
  }

  keys[index] = key;
  values[index] = value;
}

int main(int argc, const char** argv)
{
	//Initializing the Send-Receive Architecture
	WSADATA wsa;
    SOCKET s;
    struct sockaddr_in server;
    printf("-------------------------------------------------------------------------------\n");
	printf("			  Establishing Connection			\n");
	printf("-------------------------------------------------------------------------------\n");

    printf("Initialising Winsock...");
    if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
    {
        printf("Failed. Error Code : %d",WSAGetLastError());
        return 1;
    }
    
    printf("\nInitialised\n");
 
    //Create a socket
    if((s = socket(AF_INET , SOCK_STREAM , IPPROTO_TCP )) == INVALID_SOCKET)
    {
        printf("Could not create socket : %d" , WSAGetLastError());
    }
 
    printf("Socket Created\n");

    server.sin_addr.s_addr = inet_addr("127.0.0.1");
    server.sin_family = AF_INET;
    server.sin_port = htons(8888);
	
    //Connect to remote server
    if (connect(s , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        puts("connect error");
        return 1;
    }
    
    printf("\n-------------------------------------------------------------------------------\n");
	printf("			  Connection Established			\n");
	printf("-------------------------------------------------------------------------------\n");

	const int buffSize = 1000;
	char server_reply[buffSize];
	int recv_size;

	//Reading the stored vocabulary of the features of our Database Images

	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	
	//Create SURF Feature Extractor
	Ptr<FeatureDetector> detector(new SurfFeatureDetector(300,8,4,1,1));

	//To store the keypoints that will be extracted by SURF
	vector<KeyPoint> keyPoints1;

	//Create Feature Descriptor
	Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor(300,8,4,1,1));

	//Create a nearest neighbour matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());

	//Create BOW Descriptor Extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	
	//Now set the dictionary with the vocabulary of features we have already created for database images
	bowDE.setVocabulary(dictionary);

	//To store the BOW representation of an image
	Mat BOWDescriptor1;

	//Now Loading classification file
	CvSVM SVM;
	SVM.load("classification.xml");

	//Preprocessing parameters
	
	int i1=0,j1=0;
	int largest_area1=0,largest_contour_index1=0;
	double area1=0.0;
	
	vector <vector<Point>> contours1;
	vector <Vec4i> hierarchy1;
	Rect bounding_rect1;

	//Initializing the Frequency Response Parameters

	int countInstance=0;
	float responseFreq[5];
	float freqElement;
	int instanceLimit=3;

	//Initializing the parameters to extract the unique words from an array of responses
	string response_words[100]; //It will store all the live responses
	bool response_unique[100];//It will store the boolean value for the duplicate words

	string unique_words[10];//We can have a maximum of 10 unique words
	int unique_count;//Count of unique words
	int response_count;//Count live responses


	//Initializing the Text Recognition(HASH TABLE) Architecture

	 MapADT map;
	 map.insert("WAALAIKUM-US-SALAM", "D:\\Final Year Project\\DataSet Text To Gestures\\1.jpg");
     map.insert("HOW-ARE-YOU", "D:\\Final Year Project\\DataSet Text To Gestures\\2.jpg");
     map.insert("HOW-LONG", "D:\\Final Year Project\\DataSet Text To Gestures\\3.jpg");
     map.insert("MEDICINE", "D:\\Final Year Project\\DataSet Text To Gestures\\4.jpg");
     map.insert("TAKE", "D:\\Final Year Project\\DataSet Text To Gestures\\5.jpg");
     map.insert("YES", "D:\\Final Year Project\\DataSet Text To Gestures\\6.jpg");
	 map.insert("PRESCRIPTION", "D:\\Final Year Project\\DataSet Text To Gestures\\7.jpg");

	 //Initializing the send flag for the two clusters of code. If send=1 then our live recognition will work
	 //else our Hash Table function would work.

	 //Firstly assume that we are sending data so set Send_Flag=1;
	 int Send_Flag=1;

	 //When send=1 then do live recognition, send the data to server and break the loop
	 while(1)
	 {
		 if(Send_Flag==1)
		 {
			 printf("\n\n-------------------------------------------------------------------------------\n");
			 printf("			Dumb Person Sending Mode			\n");
			 printf("-------------------------------------------------------------------------------\n");

			 for(int i=0;i<100;i++)//Initializing the boolean to be true for all the words and response array to be empty
			 {
				response_unique[i]=1;
				response_words[i]="";
			 }
			 unique_count=0;
			 response_count=0;

			 printf("\nLive Gestures Response...\n\n");
			//Now to capture video form cam we decalre variable with parameter '0' for laptop cam
			VideoCapture cam(0);

			while(1)
			{
				//Initializiing the Live Gesture Recognition(Live Video) Architecture
				Mat inputImage_cam;
				cam.read(inputImage_cam);
				imshow("Cam Input",inputImage_cam);

				string name = "tmp.jpg";
				imwrite("C:\\Users\\nexus\\Desktop\\Captured Images\\"+ name, inputImage_cam);

				//Now doing the same procedure by reading the saved image

				Mat inputImage1 = imread("C:\\Users\\nexus\\Desktop\\Captured Images\\"+ name,CV_LOAD_IMAGE_UNCHANGED);

				Mat imgYCrCb1;
				cvtColor(inputImage1,imgYCrCb1,CV_BGR2YCrCb);
		
				Mat imgThreshold1;
				inRange(imgYCrCb1,Scalar(0,133,77),Scalar(255,173,127),imgThreshold1);

				findContours(imgThreshold1,contours1,hierarchy1,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

				vector <vector<Point>> hull1(contours1.size());
	
				//Find hull for each contour
				for (j1=0; j1<contours1.size(); j1++)
				{
					convexHull(Mat(contours1[j1]), hull1[j1],false);
				}

				for (i1=0; i1<contours1.size(); i1++)
				{
					area1=contourArea(contours1[i1],false);
					if(area1>largest_area1)
					{
						largest_area1=area1;
						largest_contour_index1=i1;
					}
				}

				bounding_rect1=boundingRect(contours1[largest_contour_index1]);

				Mat imgExtract1;
				imgExtract1=inputImage1(bounding_rect1);
		
				//Now we can represent our binary extracted image

				Mat finalImage1;
				cvtColor(imgExtract1,finalImage1,CV_BGR2YCrCb);
				inRange(finalImage1,Scalar(0,133,77),Scalar(255,173,127),finalImage1);
				imshow("Final Input",finalImage1);

				i1=0,j1=0;
				largest_area1=0,largest_contour_index1=0;
				area1=0.0;

				int white_pixel_count = countNonZero(finalImage1);

				if(white_pixel_count>=5000)
				{
					//Detect Feature of each frame
					detector->detect(finalImage1 ,keyPoints1);

					//Extracting BOW descriptor from the given frame
					bowDE.compute(finalImage1 ,keyPoints1, BOWDescriptor1);

					//Predicting the Sign language
					float response1 = SVM.predict(BOWDescriptor1);

					responseFreq[countInstance] = response1;
					countInstance++;
		
					if(countInstance==instanceLimit)
					{
						countInstance=0;
						freqElement = freqElementInArray(responseFreq,instanceLimit);
						string output;
						if(freqElement==1)
							output="WHERE";
						else if(freqElement==2)
							output="WHAT";
						else if(freqElement==3)
							output="THANK YOU";
						else if(freqElement==4)
							output="YES";
						else if(freqElement==5)
							output="NO";
						else if(freqElement==6)
							output="TWO";
						else if(freqElement==7)
							output="DOCTOR";
						else if(freqElement==8)
							output="ASSALAM-O-ALAIKUM";
						else if(freqElement==9)
							output="DAY";
						else if(freqElement==10)
							output="COUGH";
						cout<<output<<endl;
						response_words[response_count]=output;
						response_count++;
					}
				}

				else if(white_pixel_count<5000)
				{
					cout<<"NO IMAGE"<<endl;
				}

				if(waitKey(15)==13)
				{
					destroyAllWindows();
					break;
				}
			}

			cout<<"\n\nText Sent: ";
			
			//Firstly setting the bool value to be false for duplicate elements
			for(int i=0;i<response_count;i++)
			{
				if(response_unique[i]==1)
				{
					for(int j=i+1;j<response_count;j++)
					{
						if(strcmp(response_words[i].c_str(),response_words[j].c_str())==0)
						{
							response_unique[j]=0;
						}
					}
				}
			}

			//Now we expel the duplicate elements and send only unique words
			for(int i=0;i<response_count;i++)
			{
				if(response_unique[i]==1)
				{
					cout<<response_words[i]<<" ";
		
					if(send(s ,response_words[i].c_str(), strlen(response_words[i].c_str()) , 0) < 0)
					{
						puts("Send failed");
						return 1;
					}
					Sleep(200);
				}
			}

			Sleep(200);
			
			string terminate_cond = ".";
			if(send(s ,terminate_cond .c_str(), strlen(terminate_cond.c_str()) , 0) < 0)
			{
				puts("Send failed");
				return 1;
			}
			Send_Flag=0;
		}
		
		if(Send_Flag==0)
		{
			 printf("\n\n-------------------------------------------------------------------------------\n");
			 printf("			Dumb Person Receiving Mode			\n");
			 printf("-------------------------------------------------------------------------------\n");
			
			 printf("\nReceived Text: ");
			while((recv_size = recv(s , server_reply , buffSize , 0)) != SOCKET_ERROR)
			{
				server_reply[recv_size] = '\0';
				printf(server_reply);
				printf(" ");
				string reply=server_reply;
				if(strcmp(reply.c_str(),".")==0)
				{
					break;
				}
				Mat imgReal=imread(map.find(server_reply),CV_LOAD_IMAGE_UNCHANGED);
				namedWindow(reply,CV_WINDOW_NORMAL);
				resizeImage(reply, imgReal);
				imshow(reply,imgReal);
				waitKey(200);
			}
			Send_Flag=1;
		}
	}

	closesocket(s);
	WSACleanup();
	destroyAllWindows();
	waitKey(0);
	return 0;
}
