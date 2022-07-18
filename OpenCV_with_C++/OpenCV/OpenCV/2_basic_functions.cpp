#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Basic Functions
int main() {

	string path = "Resources/puppy.png";
	Mat img = imread(path);
	Mat imgGray, imgBlur, imgCanny, imgDil , imgErode;


	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(img, imgBlur, Size(3,3), 3, 0);
	Canny(imgBlur, imgCanny, 125, 130); //25 , 75

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);
	erode(imgDil, imgErode, kernel);

	imshow("Image", img);
	imwrite("Assets/img.png", img);
	imshow("Image Gray", imgGray);
	imwrite("Assets/imgGray.png", imgGray);
	imshow("Image Blur", imgBlur);
	imwrite("Assets/imgBlur.png", imgBlur);
	imshow("Image Canny", imgCanny);
	imwrite("Assets/imgCanny.png", imgCanny);
	imshow("Image Dilate", imgDil);
	imwrite("Assets/imgDil.png", imgDil);
	imshow("Image Erode", imgErode);
	imwrite("Assets/imgErode.png", imgErode);

	waitKey(0);

	return 0;
}