#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Resize and Crop
int main() {

	string path = "Resources/puppy.png";
	Mat img = imread(path);
	Mat imgResize, imgCrop;

	//cout << img.size() << endl;
	resize(img, imgResize, Size(), 0.5, 0.5);

	Rect roi(350, 100, 405, 575);
	imgCrop = img(roi);


	imshow("Image", img);
	imshow("Image Resize", imgResize);
	imwrite("Assets/imgResize.png", imgResize);
	imshow("Image Crop", imgCrop);
	imwrite("Assets/imgCrop.png", imgCrop);

	waitKey(0);

	return 0;
}