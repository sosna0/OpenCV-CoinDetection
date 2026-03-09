#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "CoinDetector.h"
using namespace cv;


std::string windowName = "coinDetection";

int main(int argc, char* argv[]) {
   
    
    if (argc < 2) {
        std::cout << "You have to provide path for the image" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];

    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()){
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    resize(img, img, Size(0,0), 0.35, 0.35);

    namedWindow(windowName);
   
    CoinDetector coinDetector(img, windowName);

    while (true) {
        int k = waitKey(10);
        // 27 - escape key
        if (k == 27) {
            break;
        }

        if (getWindowProperty(windowName, WND_PROP_VISIBLE) < 1) {
            break;
        }

        auto [imgLabeled, imgContours] = coinDetector.Update();

        Mat imgContoursBGR;
        cvtColor(imgContours, imgContoursBGR, COLOR_GRAY2BGR);

        Mat concatImages;
        hconcat(imgLabeled, imgContoursBGR, concatImages);
        
        imshow(windowName, concatImages);
    }

    destroyAllWindows();

    return 0;
}