#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>

#define USE_CUDA true //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;


int main()
{
	string img_path = "./images/bus.jpg";

#if(defined YOLOV5 && YOLOV5==true)
	string model_path = "models/yolov5s.onnx";
#else
	string model_path = "models/yolov7.onnx";
#endif


	Yolo test;
	Net net;
	if (test.readModel(net, model_path, USE_CUDA)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}

	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<Output> result;
	Mat img = imread(img_path);

	if (test.Detect(img, net, result)) {
		test.drawPred(img, result, color);

	}
	else {
		cout << "Detect Failed!" << endl;
	}

	system("pause");
	return 0;
}
