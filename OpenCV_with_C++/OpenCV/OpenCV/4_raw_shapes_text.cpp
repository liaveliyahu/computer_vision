#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Draw Shapes and Text
int main() {

	// Blank Image
	Mat img(512, 512, CV_8UC3, Scalar(255,255, 255));

	
	rectangle(img, Point(50, 200), Point(450, 300), Scalar(0, 216, 255), FILLED);

	rectangle(img, Point(58, 208), Point(147, 247), Scalar(0, 0, 0), FILLED);
	rectangle(img, Point(60, 210), Point(145, 245), Scalar(255, 255, 255), FILLED);

	rectangle(img, Point(143, 208), Point(232, 247), Scalar(0, 0, 0), FILLED);
	rectangle(img, Point(145, 210), Point(230, 245), Scalar(255, 255, 255), FILLED);

	rectangle(img, Point(228, 208), Point(317, 247), Scalar(0, 0, 0), FILLED);
	rectangle(img, Point(230, 210), Point(315, 245), Scalar(255, 255, 255), FILLED);

	rectangle(img, Point(315, 208), Point(352, 300), Scalar(0, 0, 0), FILLED);
	rectangle(img, Point(317, 210), Point(350, 298), Scalar(255, 255, 255), FILLED);

	rectangle(img, Point(352, 208), Point(442, 247), Scalar(0, 0, 0), FILLED);
	rectangle(img, Point(354, 210), Point(440, 245), Scalar(255, 255, 255), FILLED);

	vector<vector<Point>> contours1{ { Point(380, 247) , Point(442, 247) , Point(442, 275)} };
	drawContours(img, contours1, 0, Scalar(0, 0, 0), -1);

	vector<vector<Point>> contours2{ { Point(386, 248) , Point(440, 248) , Point(440, 272)} };
	drawContours(img, contours2, 0, Scalar(255, 255, 255), -1);

	circle(img, Point(120, 300), 20, Scalar(0, 0, 0), FILLED);
	circle(img, Point(380, 300), 20, Scalar(0, 0, 0), FILLED);
	circle(img, Point(120, 300), 12, Scalar(255, 255, 255), FILLED);
	circle(img, Point(380, 300), 12, Scalar(255, 255, 255), FILLED);
	circle(img, Point(120, 300), 8, Scalar(0, 0, 0), FILLED);
	circle(img, Point(380, 300), 8, Scalar(0, 0, 0), FILLED);

	putText(img, "Computer Vision", Point(70, 274), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 0), 2);
	putText(img, "Computer Vision", Point(72, 272), FONT_HERSHEY_COMPLEX, 0.8, Scalar(30, 14, 150), 2);
	

	imshow("Image", img);
	imwrite("Assets/cv_bus.png", img);

	waitKey(0);

	return 0;

}