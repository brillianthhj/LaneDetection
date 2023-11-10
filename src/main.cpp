#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

enum Direction : uint8_t
{
	// lane direction
	LEFT = 0,
	RIGHT = 1
};

int32_t findEdges(const cv::Mat& img, Direction direction);
void drawCross(cv::Mat& img, cv::Point pt, cv::Scalar color);

int main()
{
    cv::VideoCapture cap;
    bool flag = cap.open("/home/hyejin/Playground/LaneDetection/resource/SubProject01.avi");

	if (!cap.isOpened())
	{
		std::cout << "Can't open video!!" << std::endl;
		return -1;
	}

	int32_t width = cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int32_t height = cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    //int  fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
    int32_t  fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size size(width, height);

    cv::VideoWriter output("/home/hyejin/Playground/LaneDetection/output01.avi", fourcc, fps, size);

	int32_t delay = cvRound(1000 / fps);
	cv::Mat frame, dx, dy, edge, gray;

	// csv writer setting
	std::fstream fout;
	fout.open("/home/hyejin/Playground/LaneDetection/coordinates.csv", std::ios::out);
	fout << "#coord_left" << "," << "#coord_right" << "\n";
	
	constexpr int32_t offset = 400;
	constexpr int32_t rectHeight = 20;
	const auto rectWidth = static_cast<int32_t>(width * 0.5);
	constexpr int32_t padding = 12;

	// left, right rectangles
	cv::Rect rectLeft(0, offset-10, rectWidth, rectHeight);
	cv::Rect rectRight(rectWidth, offset-10, rectWidth, rectHeight);

	// kalman filter configs
	cv::KalmanFilter KF(4, 2, 0);
    // cv::Mat state(4, 1, CV_32F);
    cv::Mat processNoise(4, 1, CV_32F);
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
	cv::Mat improved(4, 1, CV_32F);

	KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, cv::Scalar::all(1));
	randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));

	bool first = true;

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		std::cout << "frame is running." << std::endl;

		// get points
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		int32_t ptsLeft = findEdges(gray(rectLeft), Direction::LEFT) - padding;
		int32_t ptsRight = findEdges(gray(rectRight), Direction::RIGHT) + rectWidth + padding;

/*
		// if (first) 
		// {
		state.at<float>(0) = ptsLeft;
		state.at<float>(1) = ptsRight;
		state.at<float>(2) = 0.f;
		state.at<float>(3) = 0.f;
			first = false;
		// }

		// kalman pred
		// cv::Mat prediction = KF.predict();

		// generate measurement
		// randn(measurement, cv::Scalar::all(0), cv::Scalar::all(KF.measurementNoiseCov.at<float>(0)));
		// measurement += KF.measurementMatrix*state;

		// correct the state estimates based on measurements
		// updates statePost & errorCovPost
		KF.correct(measurement);
		cv::Mat improved = KF.statePost;

		// forecast point
		cv::Mat forecast = KF.transitionMatrix*KF.statePost;
*/

		// predict
		if (ptsLeft < 0)
			ptsLeft = improved.at<float>(0);
		
		if (ptsRight > 640)
			ptsRight = improved.at<float>(1);
		
		measurement.at<float>(0) = ptsLeft;
		measurement.at<float>(1) = ptsRight;

		KF.correct(measurement);
		improved = KF.predict();
		//improved = KF.transitionMatrix*KF.statePost;


		// draw left cross
		int32_t xLeft = (ptsLeft > 0) ? ptsLeft : 0;
		drawCross(frame, cv::Point(xLeft, offset), cv::Scalar(0, 0, 255));
		putText(frame, cv::format("(%d, %d)", xLeft, offset),
			cv::Point(xLeft - 50, offset - 20),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		// draw right cross
		int32_t xRight = (ptsRight < 640) ? ptsRight : 640;
		drawCross(frame, cv::Point(xRight, offset), cv::Scalar(0, 0, 255));
		putText(frame, cv::format("(%d, %d)", xRight, offset),
			cv::Point(xRight - 50, offset - 20),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		// draw ractangle
		rectangle(frame, cv::Rect(0, offset-10, rectWidth * 2, rectHeight), cv::Scalar(0, 0, 255), 2);
		
		// write csv
		fout << xLeft << "," << xRight << "\n";

		output << frame;

		// update kalman state
		// randn( processNoise, cv::Scalar(0), cv::Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
		// state = KF.transitionMatrix*state + processNoise;
	}
	fout.close();
	std::cout << "out of bracket" << std::endl;

	output.release();
	cap.release();
  
}

int32_t findEdges(const cv::Mat& img, Direction direction)
{
	cv::Mat fimg, blr, dy;
	img.convertTo(fimg, CV_32F);
	GaussianBlur(fimg, blr, cv::Size(), 1.);
	Sobel(blr, dy, CV_32F, 0, 1);
	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
	morphologyEx(dy, dy, cv::MORPH_CLOSE, kernel);

	double maxValue;
	cv::Point maxLoc;

	int32_t halfY = fimg.rows/2;
	cv::Mat roi = dy.row(halfY);
	minMaxLoc(roi, NULL, &maxValue, NULL, &maxLoc);

	int32_t threshold = 90;
	int32_t xCoord = (maxValue > threshold) ? maxLoc.x : (direction == Direction::LEFT) ? 0 : 320;

	return xCoord;
}

void drawCross(cv::Mat& img, cv::Point pt, cv::Scalar color)
{
	int32_t span = 5;
	line(img, pt + cv::Point(-span, -span), pt + cv::Point(span, span), color, 1, cv::LINE_AA);
	line(img, pt + cv::Point(-span, span), pt + cv::Point(span, -span), color, 1, cv::LINE_AA);
}