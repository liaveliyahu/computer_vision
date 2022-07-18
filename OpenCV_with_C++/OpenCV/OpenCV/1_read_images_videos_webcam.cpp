#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//// Images
//int main() {
//
//	string path = "Resources/puppy.png";
//	Mat img = imread(path);
//	imshow("Image", img);
//	waitKey(0);
// 
//  return 0;
//
//}


//// Video
//int main() {
//
//	string path = "Resources/puppies.mp4";
//	VideoCapture cap(path);
//	Mat img;
//
//	while (true) {
//
//		cap.read(img);
//		imshow("Video", img);
//		waitKey(20);
//
//	}
// 
//  return 0;
//
//}

// Webcam
int main() {

	VideoCapture cap(0);
	Mat img;

	while (true) {

		cap.read(img);
		imshow("Webcam", img);
		waitKey(1);

	}

	return 0;
}