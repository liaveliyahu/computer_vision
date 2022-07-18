#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// Images
int main() {

	string path = "Resources/people.png";
	Mat img = imread(path);

	CascadeClassifier faceCascade;
	faceCascade.load("Resources/haarcascade_frontalface_default.xml");

	if (faceCascade.empty())
		cout << "XML file not loaded" << endl;

	vector<Rect> faces;
	faceCascade.detectMultiScale(img, faces, 1.3, 10);

	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
	}

	imshow("Image", img);
	imwrite("Assets/detected_faces.png", img);
	waitKey(0);

	return 0;

}

//// Webcam
//int main() {
//
//	VideoCapture cap(0);
//	Mat img;
//
//	while (true) {
//
//		cap.read(img);
//
//
//		CascadeClassifier faceCascade;
//		faceCascade.load("Resources/haarcascade_frontalface_default.xml");
//
//		if (faceCascade.empty())
//			cout << "XML file not loaded" << endl;
//
//		vector<Rect> faces;
//		faceCascade.detectMultiScale(img, faces, 1.1, 10);
//
//		for (int i = 0; i < faces.size(); i++)
//		{
//			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
//		}
//
//		imshow("Image", img);
//		waitKey(1);
//
//	}
//	return 0;
//}