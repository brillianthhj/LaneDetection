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

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		std::cout << "frame is running." << std::endl;

		// get points
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		int32_t ptsLeft = findEdges(gray(rectLeft), Direction::LEFT);
		int32_t ptsRight = findEdges(gray(rectRight), Direction::RIGHT);

		// draw left cross
		int32_t xLeft = (ptsLeft - padding >= 0) ? ptsLeft - padding : 0;
		drawCross(frame, cv::Point(xLeft, offset), cv::Scalar(0, 0, 255));
		putText(frame, cv::format("(%d, %d)", xLeft, offset),
			cv::Point(ptsLeft - 50, offset - 20),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		// draw right cross
		int32_t xRight = (rectWidth + ptsRight + padding <= 640) ? rectWidth + ptsRight + padding : 640;
		drawCross(frame, cv::Point(xRight, offset), cv::Scalar(0, 0, 255));
		putText(frame, cv::format("(%d, %d)", xRight, offset),
			cv::Point(rectWidth + ptsRight - 50, offset - 20),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		// draw ractangle
		rectangle(frame, cv::Rect(0, offset-10, rectWidth * 2, rectHeight), cv::Scalar(0, 0, 255), 2);
		
		// write csv
		fout << xLeft << "," << xRight << "\n";

		output << frame;
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
