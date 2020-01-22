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


#include "iostream"
#include "cmath"
using namespace cv;
using namespace std;

//To use SURF Featurer Detector add the library #include "opencv2/nonfree/features2d.hpp"
//and also add additional depndency opencv_nonfree2410d.lib and opencv_flann2410d for FLANN Matcher

Mat PreProcessed_Image(String inputImage)
{
	int i,j;
	int largest_area=0,largest_contour_index=0;
	double area;
	Vec3b int_YCrCb,int_RGB;
	vector <vector<Point>> contours;
	vector <Vec4i> hierarchy;
	Rect bounding_rect;

	Mat imgRGB;
	imgRGB=imread(inputImage,CV_LOAD_IMAGE_UNCHANGED); //Reading Input Image for Preprocessing

	Mat imgYCrCb;
	cvtColor(imgRGB,imgYCrCb,CV_BGR2YCrCb);		//Converting color-space from RGB to YCrCb

	Mat imgThreshold;
	inRange(imgYCrCb,Scalar(0,133,77),Scalar(255,173,127),imgThreshold);	//Setting threshold according to skin-color range to detect hand
	
	findContours(imgThreshold,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);  //Finding all the  contours according  to the thresholds set above

	vector <vector<Point>> hull(contours.size());
	
	//Find hull for each contour
	for (j=0; j<contours.size(); j++)
	{
		convexHull(Mat(contours[j]), hull[j],false);
	}

	for (i=0; i<contours.size(); i++)
	{
		area=contourArea(contours[i],false); //Finding area of each contour
		if(area>largest_area)	//Finding the contour having the largest area assuming its a hand
		{
			largest_area=area;
			largest_contour_index=i;
		}
	}

	bounding_rect=boundingRect(contours[largest_contour_index]); //Making a bounding rectangle around the hand

	Mat imgExtract;
	imgExtract=imgRGB(bounding_rect);  //Extracting image of hand from the entire image

	//Now we can represent our binary extracted image

	Mat finalImage;
	cvtColor(imgExtract,finalImage,CV_BGR2YCrCb);
	inRange(finalImage,Scalar(0,133,77),Scalar(255,173,127),finalImage);

	return finalImage;

}

int main(int argc, const char** argv)
{
	//Input image from database
	Mat inputImage;

	//String for a specific image in database
	int totalImages=1000;
	char * imageDatabase = new char[totalImages];

	//Reading the stored vocabulary of the features of our Database Images

	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	//Create SURF Feature Extractor
	Ptr<FeatureDetector> detector(new SurfFeatureDetector(300,8,4,1,1));

	//To store the keypoints that will be extracted by SURF
	vector<KeyPoint> keyPoints;

	//Create Feature Descriptor
	Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor(300,8,4,1,1));

	//Create a nearest neighbour matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());

	//Create BOW Descriptor Extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	
	//Now set the dictionary with the vocabulary of features we have already created for database images
	bowDE.setVocabulary(dictionary);

	//To store the BOW representation of an image
	Mat BOWDescriptor;

	Mat groundTruth(0,1,CV_32FC1);
	Mat evalData(0,10,CV_32FC1);
	Mat results(0,1,CV_32FC1);
	
	//Now Loading classification file
	CvSVM SVM;
	SVM.load("classification.xml");
		
	for(int i=1;i<=10;i++)
	{
		for(int j=1;j<=15;j++)
		{
			sprintf(imageDatabase,"D:\\Final Year Project\\DataSet Same Background\\Data Set Scenario\\Testing Data Set\\%d (%d).jpg",i,j);
			inputImage = imread(imageDatabase);
			//imshow("Input Image",inputImage);
			inputImage = PreProcessed_Image(imageDatabase);
			//imshow("Filtered Image",inputImage);

			//Detect Feature of each image
			detector->detect(inputImage,keyPoints);

			//Extracting BOW descriptor from the given image
			bowDE.compute(inputImage,keyPoints,BOWDescriptor);
			
			//Updating Train Data
			evalData.push_back(BOWDescriptor);
			
			//Updating Response data for each class like 1,2,3 for three different classes of each number 1,2 and 3 etc.
			groundTruth.push_back((float)i);
			float response = SVM.predict(BOWDescriptor);
			cout<<i<<"("<<j<<")		"<<response<<endl;
			results.push_back(response);
		}
	}
	
	double error_rate = (double) countNonZero(groundTruth-results)/evalData.rows;
	cout<<"\n\nAccuracy Rate: "<<(1-error_rate)*100<<endl;
	imshow("Input Image", inputImage);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
