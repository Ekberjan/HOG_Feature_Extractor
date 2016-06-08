// Makeup.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "hog.h"


int _tmain(int argc, _TCHAR* argv[])
{
	cv::Mat image = cv::imread("/Images/1.png");
	cv::cvtColor(image, image, CV_BGR2GRAY);
	
	cv::Mat source, gaborOutput;
	image.convertTo(source, CV_32F);

	// Get HOG features
	HOG myHog;
	myHog.run(image);
	cv::vector<cv::Mat> hogFeature = myHog.getFeatureVec();
	
	std::vector<double> feature; // vector for storing hog features
	
	// convert the cv::Mat type members into vector of doubles
	for (int x = 0; x < hogFeature.size(); x++)
	{
		cv::Mat hogMat = hogFeature[x];
		for (int y = 0; y < hogMat.rows; y++)
		{
			const double* My = hogMat.ptr<double>(y);

			for (int z = 0; z < hogMat.cols; z++)
			{
				feature.push_back(My[z]);
			} // end of z-loop
		} // end of y-loop
	} // end of x-loop
	
	std::cout << "feature size: " << feature.size() << std::endl;

	// Use the obtained HOG features for further steps, 
	// such as send it to classifier.


	return 0;
}

