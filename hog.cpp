#include "stdafx.h"
#include <iostream>
#include "hog.h"

HOG::HOG()
{
	nwin_x = 3;
	nwin_y = 3;
	B = 9;
}

void HOG::run(cv::Mat & inputImage)
{
	cv::Mat image = inputImage;
	int width = image.cols / 5;
	int height = image.rows / 5;

	for (int i = 0; i < image.rows; i += height)
	{
		for (int j = 0; j < image.cols; j += width)
		{
			int rowx = i + height;
			int coly = j + width;

			if (rowx > image.rows)
			{
				rowx = image.rows;
			}

			if (coly > image.cols)
			{
				coly = image.cols;
			}

			cv::Mat subImage = image.rowRange(i, i + 5).colRange(j, j + 5);
			cv::Mat hogFeature = getFeature(subImage);
			hogFeatureVec.push_back(hogFeature);

		} // end of j-loop
	} // end of i-loop


} // end of function definition

cv::Mat HOG::getFeature(cv::Mat & inputImage)
{
	cv::Mat img = inputImage;
	// check if input is grayscale
	if (img.channels() > 1)
	{
		cv::cvtColor(img, img, CV_BGR2GRAY);
	}

	img.convertTo(img, CV_32FC1);

	int L = img.rows;
	int C = img.cols;

	cv::Mat H = cv::Mat::zeros(nwin_x * nwin_y * B, 1, CV_32F);
	double m = std::sqrt(L / 2);

	if (C == 1) // if number of column equals 1
	{
		cv::resize(img, img, cv::Size(m, 2 * m));
		L = 2 * m;
		C = m;
	}

	int step_x = std::floor(C / (nwin_x + 1));
	int step_y = std::floor(L / (nwin_y + 1));
	int cont = 0;

	cv::Mat hx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
	cv::Mat hy = -hx.t();

	// matrices for gradients of x and y directions
	cv::Mat grad_xr, grad_yu;

	// get gradients of x and y directions
	cv::filter2D(img, grad_xr, -1, hx);
	cv::filter2D(img, grad_yu, -1, hy);

	cv::Mat angles = cv::Mat::zeros(L, C, CV_32FC1);
	cv::Mat magnit = cv::Mat::zeros(L, C, CV_32FC1);

	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < C; j++)
		{
			float pixx = grad_xr.at<float>(i, j);
			float pixy = grad_yu.at<float>(i, j);
			float pix = std::atan2(pixy, pixx);
			angles.at<float>(i, j) = pix;

			float pix2 = pixy * pixy + pixx * pixx;
			pix2 = std::pow(pix2, 0.5);
			magnit.at<float>(i, j) = pix2;

			//std::cout << "pix2: " << pix2 << std::endl;

		} // end of j-loop
	} // end of i-loop
		
	magnit /= 255.0;

	for (int n = 0; n < nwin_y; n++)
	{
		for (int m = 0; m < nwin_x; m++)
		{
			cv::Mat angles2 = angles.rowRange(n * step_y, (n + 2) * step_y).colRange(m * step_x, (m + 2) * step_x);
			cv::Mat magnit2 = magnit.rowRange(n * step_y, (n + 2) * step_y).colRange(m * step_x, (m + 2) * step_x);

			cv::Mat v_angles = cv::Mat::zeros(L * C, 1, CV_32FC1);
			cv::Mat v_magnit = cv::Mat::zeros(L * C, 1, CV_32FC1);

			for (int x = 0; x < angles2.cols; x++)
			{
				cv::Mat v_angles2 = v_angles(cv::Rect(0, x * L, 1, angles2.col(x).rows));
				cv::Mat v_magnit2 = v_magnit(cv::Rect(0, x * L, 1, magnit2.col(x).rows));

				angles2.col(x).copyTo(v_angles2);
				magnit2.col(x).copyTo(v_magnit2);

			} // end of x-loop
			
			int K = std::max(v_angles.rows, v_angles.cols);
			
			// assembling the histogram with 9 bins (range of 20 degrees per bin)
			int bin = 0;
			cv::Mat H2 = cv::Mat::zeros(B, 1, CV_32FC1);

			for (float ang_lim = -PI + 2 * PI / B; ang_lim <= PI; ang_lim += 2 * PI / B)
			{
				for (int k = 0; k < K; k++)
				{
					if (v_angles.at<float>(k, 0) < ang_lim)
					{
						v_angles.at<float>(k, 0) = 100;
						H2.at<float>(bin, 0) = H2.at<float>(bin, 0) + v_magnit.at<float>(k, 0);
					}
				} // end of k-loop

				bin++;
				
			} // end of ang_lim-loop

			H2 /= cv::norm(H2) + 0.01;
			cv::Mat tmpH = H(cv::Rect(0, cont*B, 1, H2.rows));
			H2.copyTo(tmpH);
			cont++;

		} // end of m-loop
	} // end of n-loop

	return H;
} // end of function definition

cv::vector<cv::Mat> HOG::getFeatureVec()
{
	return hogFeatureVec;
}