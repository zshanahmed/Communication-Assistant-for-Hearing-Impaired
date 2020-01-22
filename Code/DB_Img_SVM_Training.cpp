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
#include "math.h"
using namespace cv;
using namespace std;

//To use SURF Featurer Detector add the library #include "opencv2/nonfree/features2d.hpp"
//and also add additional depndency opencv_nonfree2410d.lib and opencv_flann2410d for FLANN Matcher

int main(int argc, const char** argv)
{
	//String for a specific image in database
	int totalImages = 1000;
	char * imageDatabase = new char[totalImages];

	//Input image from database
	Mat inputImage;

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

	//Now initializing two matrices for trainingdata and their responses for SVM classification
	Mat trainingData(0,5000,CV_32FC1);
	Mat labels(0, 1, CV_32FC1);

	//Extracting histogram in the form of BoW for each image
	for(int i=1;i<=10;i++)
	{
		for(int j=1;j<=100;j++)
		{
			sprintf(imageDatabase,"D:\\Final Year Project\\DataSet Preprocessed\\Data Set Scenario\\%d (%d).jpg",i,j);
			inputImage = imread(imageDatabase);

			//Detect Feature of each image
			detector->detect(inputImage,keyPoints);

			//Extracting BOW descriptor from the given image
			bowDE.compute(inputImage,keyPoints,BOWDescriptor);

			//Updating Train Data
			trainingData.push_back(BOWDescriptor);
					
			//Updating Response data for each class like 1,2,3 for three different classes of each number 1,2 and 3 etc.
			labels.push_back((float)i);
		}
		cout<<i<<endl;	
}

	cout<<"Data Collection for Training Completed"<<endl;
	cout<<"Now training SVM Classifier......."<<endl;

	CvParamGrid CvParamGrid_C(1, 1000, 1.2);
	CvParamGrid CvParamGrid_gamma(0.1,100,1.2);
	
	//Defining SVM parameters
	
	CvSVMParams svm_params;
	svm_params.kernel_type = CvSVM::RBF;
	svm_params.svm_type = CvSVM::C_SVC;
	svm_params.gamma = 4.1018627024600171e+001;
	svm_params.C = 7.4300837068799988e+000;
	svm_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	
	//Training the SVM

	CvSVM SVM;
	SVM.train(trainingData,labels,Mat(),Mat(),svm_params);

	svm_params=SVM.get_params();

	cout<<"C: "<<svm_params.C<<endl;
	cout<<"Gamma: "<<svm_params.gamma<<endl;

	SVM.save("classification.xml");

	Mat image = imread("D:\\Final Year Project\\DataSet Preprocessed\\Data Set Scenario\\1 (1).jpg",CV_LOAD_IMAGE_UNCHANGED);
	imshow("Image",image);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
