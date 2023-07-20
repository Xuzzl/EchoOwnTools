//#include <windows.h>
//#include<iostream>
//#include <opencv2/opencv.hpp>
//#include"ColorDetector.h"
//
//using namespace cv;
//using namespace std;
//
//void onchange(int pos, void* data);
//void onchange_r(int pos, void* data);
//void onchange_g(int pos, void* data);
//void onchange_b(int pos, void* data);
//int main()
//{
//	ColorDetector* CD = ColorDetector::getInstance();
//	CD->setInputImage("D://code//EchoOwnTools//color.png");
//	//Mat img = CD->getInputImage();
//	CD->setTargetColor(0, 0, 255);
//	CD->setColorDistanceThreshold(10);
//	CD->process();
//	Mat out = CD->getResult();
//	namedWindow("窗口", WINDOW_NORMAL);
//	resizeWindow("窗口", 600, 600);
//	imshow("窗口", out);
//
//	int tb_value = CD->getColorDistanceThreshold();
//	Vec3b RGB_value = CD->getTargetColor();
//	int R_value = (int)RGB_value[0];
//	int G_value = (int)RGB_value[1];
//	int B_value = (int)RGB_value[2];
//	namedWindow("窗口1");
//	resizeWindow("窗口1", 600, 300);
//	createTrackbar("阈值", "窗口1", &tb_value, 1000, onchange, CD);
//	createTrackbar("R", "窗口1", &R_value, 255, onchange_r, CD);
//	createTrackbar("G", "窗口1", &G_value, 255, onchange_g, CD);
//	createTrackbar("B", "窗口1", &B_value, 255, onchange_b, CD);
//	waitKey();
//	return 0;
//}
//void onchange(int pos, void* data)
//{
//	ColorDetector* CD = (ColorDetector*)data;
//	CD->setColorDistanceThreshold(pos);
//	CD->process();
//	imshow("窗口", CD->getResult());
//}
//void onchange_r(int pos, void* data)
//{
//	ColorDetector* CD = (ColorDetector*)data;
//	Vec3b RGB_value = CD->getTargetColor();
//	RGB_value[2] = (uchar)pos;
//	CD->setTargetColor(RGB_value);
//	CD->process();
//	imshow("窗口", CD->getResult());
//}
//void onchange_g(int pos, void* data)
//{
//	ColorDetector* CD = (ColorDetector*)data;
//	Vec3b RGB_value = CD->getTargetColor();
//	RGB_value[1] = (uchar)pos;
//	CD->setTargetColor(RGB_value);
//	CD->process();
//	imshow("窗口", CD->getResult());
//}
//void onchange_b(int pos, void* data)
//{
//	ColorDetector* CD = (ColorDetector*)data;
//	Vec3b RGB_value = CD->getTargetColor();
//	RGB_value[0] = (uchar)pos;
//	CD->setTargetColor(RGB_value);
//	CD->process();
//	imshow("窗口", CD->getResult());
//}