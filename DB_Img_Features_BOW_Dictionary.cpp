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

int main(int argc, const char** argv)
{
	int totalImages = 1000;
	
	//To store the Database images Names
	char * imageDatabase = new char[totalImages];

	//To store the Database image after segmentation and binarization
	Mat inputImage;
	
	//To detect SURF Features of the inputImage
	SurfFeatureDetector detector(300,8,3,1,1);

	//To store the keypoints extracted by SURF feature Detector
	vector<KeyPoint> keyPoints;

	//To extract descriptors from each of SURF Features
	SurfDescriptorExtractor extractor(300,8,3,1,1);

	//To store the descriptor of each feature
	Mat descriptor;

	//Constructing BOWKMeans Trainer to cluster the descriptors
	
	int dictionarySize = 5000;
	TermCriteria tc(CV_TERMCRIT_ITER,1000,0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,KMEANS_PP_CENTERS);

	for(int i=1;i<=10;i++)
	{
		for(int j=1;j<=100;j++)
		{
			sprintf(imageDatabase,"D:\\Final Year Project\\DataSet Preprocessed\\Data Set Scenario\\%d (%d).jpg",i,j);			
			inputImage = imread(imageDatabase);

			//Detect Feature of each image
			detector.detect(inputImage,keyPoints);
		
			//Extract descriptor of the image
			extractor.compute(inputImage,keyPoints,descriptor);
			
			//Put the descriptor in a single object
			bowTrainer.add(descriptor);
		}
		cout<<i<<endl;
	}

	cout<<"Features Collection Done"<<endl;
	
	//Cluster the feature Vectors
	Mat dictionary = bowTrainer.cluster();

	cout<<"Vocabulary Building Done"<<endl;
	
	//Store The vocabulary
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	waitKey(0);
	destroyAllWindows();
	return 0;
}