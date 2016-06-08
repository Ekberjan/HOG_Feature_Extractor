#ifndef HOG_H
#define HOG_H

#define PI 3.14159

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

class HOG
{
public:
	HOG();	
	void run(cv::Mat & inputImage);
	cv::vector<cv::Mat> getFeatureVec();
		
private:
	int nwin_x, nwin_y, B;	
	cv::vector<cv::Mat> hogFeatureVec;
	cv::Mat getFeature(cv::Mat & inputImage);

};


#endif