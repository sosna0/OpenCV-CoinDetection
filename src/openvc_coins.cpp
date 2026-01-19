//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <print>
#include <vector>
#include <tuple>

using namespace cv;
using Contours = std::vector<std::vector<Point>>;
using Hierarchy = std::vector<Vec4i>;

const Scalar GREEN = Scalar(0, 255, 0);

static void on_trackbar(int pos, void *) { std::println("{0}", pos); return; }

struct HSV{
    int hmin;
    int smin;
    int vmin;

    int hmax;
    int smax;
    int vmax;
};

class ColorFinder {
public:
    ColorFinder(bool trackbars = false) : _trackbars(trackbars) {
        if (trackbars) {
            initTrackbars();
        }
    }
    ~ColorFinder(){}

    void initTrackbars() {
        namedWindow("TrackBars");
        resizeWindow("TrackBars", 640, 240);
        createTrackbar("Hue Min", "TrackBars", nullptr, 179, on_trackbar);
        createTrackbar("Hue Max", "TrackBars", nullptr, 179, on_trackbar);
        createTrackbar("Sat Min", "TrackBars", nullptr, 255, on_trackbar);
        createTrackbar("Sat Max", "TrackBars", nullptr, 255, on_trackbar);
        createTrackbar("Val Min", "TrackBars", nullptr, 255, on_trackbar);
        createTrackbar("Val Max", "TrackBars", nullptr, 255, on_trackbar);
    }

    HSV getTrackbarValues() {
        
        HSV hsv;

        hsv.hmin = getTrackbarPos("Hue Min", "TrackBars");
        hsv.smin = getTrackbarPos("Sat Min", "TrackBars");
        hsv.vmin = getTrackbarPos("Val Min", "TrackBars");

        hsv.hmax = getTrackbarPos("Hue Max", "TrackBars");
        hsv.smax = getTrackbarPos("Sat Max", "TrackBars");
        hsv.vmax = getTrackbarPos("Val Max", "TrackBars");

        return hsv;
    }

    std::tuple<Mat, Mat> update(cv::Mat img, HSV *myHSV = nullptr) {
        cv::Mat imgColor;
        cv::Mat mask;
        HSV tmp{0};

        if (this->_trackbars) {
            if (myHSV == nullptr) {
                myHSV = &tmp;
            }
            *myHSV = this->getTrackbarValues();    
        }
        
        if (myHSV != nullptr) {
            cv::Mat imgHSV;
            cvtColor(img, imgHSV, COLOR_BGR2HSV);
            
            std::vector<int> lower = { myHSV->hmin, myHSV->smin, myHSV->vmin };
            std::vector<int> upper = { myHSV->hmax, myHSV->smax, myHSV->vmax };
            
            cv::inRange(imgHSV, lower, upper, mask);
            cv::bitwise_and(img, img, imgColor, mask);
        }
               
        return { imgColor, mask };
    }



private:
    bool _trackbars;
};



std::string windowName = "coinDetection";

std::string trackbarCanny1 = "CannyThr1";
std::string trackbarCanny2 = "CannyThr2";
std::string trackbarBlur = "Blur2N+1";
std::string trackbarThresh = "Threshold";



Mat preProcessing(Mat &img) {

    Mat blur_img;
    int n = 2; //getTrackbarPos(trackbarBlur, windowName);
    GaussianBlur(img, blur_img, Size(2*n+1, 2*n+1), 0);

    Mat canny_img;
    int thresh1 = getTrackbarPos(trackbarCanny1, windowName);
    int thresh2 = getTrackbarPos(trackbarCanny2, windowName);
    Canny(blur_img, canny_img, thresh1, thresh2);


    Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(3,3));
    
    Mat dialated_img;
    //int morph_oper1 = getTrackbarPos(trackbarMorph1, windowName);
    morphologyEx(canny_img, dialated_img, MorphTypes::MORPH_DILATE, kernel);

    Mat morph_close_img;
    //int morph_oper2 = getTrackbarPos(trackbarMorph2, windowName);
    morphologyEx(dialated_img, morph_close_img, MorphTypes::MORPH_CLOSE, kernel);

    return morph_close_img;
}


Contours findCircles(Contours contours, Hierarchy hierarchy){
    Contours circles;

    float epsilon = 0.15;

    for (const auto &cnt : contours) {
        double area = contourArea(cnt);
        //if (area >= minArea && area <= maxArea) {
            double peri = arcLength(cnt, true);
            std::vector<Point> approxPolyVec;
            approxPolyDP(cnt, approxPolyVec, 0.02 * peri, true);

            auto polyNum = approxPolyVec.size();
            //std::println("Polygons: [{0}]", polyNum);

            if (polyNum > 5) {
                auto rect = cv::boundingRect(cnt);
                float ratio = rect.height / static_cast<float>(rect.width);
                ratio = std::min(ratio, 1.0f / ratio);

                if (ratio >= 1.0f - epsilon) {
                    circles.emplace_back(cnt);
                }


            }
        //}
    }

    return circles;
}

/** @brief Draws bounding boxes around contours.**/
void drawBoundaries(Mat img, Contours circles) {
    
    for (const auto &circle : circles) {
        auto rect = cv::boundingRect(circle);
        rectangle(img, rect, GREEN, 2);
        
        Point center = Point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);
        cv::circle(img, center, static_cast<int>(rect.width / 2.0), GREEN, 2);
        cv::circle(img, center, 3, GREEN, FILLED);
    }
}

int main(){
    //std::string image_path = samples::findFile("images/coins2.webp");
    //std::string image_path = samples::findFile("images/coins_red_bg.jpg");
    //std::string image_path = samples::findFile("images/img1.jfif");
    //std::string image_path = samples::findFile("images/img2.jfif");
    std::string image_path = samples::findFile("images/img3.jfif");

    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()){
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    resize(img, img, Size(0,0), 0.25, 0.25); // 0.4
    
    Mat gray_img;
    Mat color_img = img.clone();
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    namedWindow(windowName);
    //createTrackbar(trackbarBlur, windowName, nullptr, 15, on_trackbar);
    //createTrackbar(trackbarThresh, windowName, nullptr, 255, on_trackbar);
    createTrackbar(trackbarCanny1, windowName, nullptr, 255, on_trackbar);
    createTrackbar(trackbarCanny2, windowName, nullptr, 255, on_trackbar);

    HSV my_hsv{
        10, 55, 215, 42, 255, 255
    };

    //ColorFinder myColorFinder = ColorFinder(true);

    while (true) {
        int k = waitKey(10);
        // 27 - escape key
        if (k == 27) {
            break;
        }

        if (getWindowProperty(windowName, WND_PROP_VISIBLE) < 1) {
            break;
        }

        Mat imgTmp = color_img.clone();
        Mat imgPre = preProcessing(gray_img);

        Contours contours;
        // info about figure topology
        Hierarchy hierarchy;
            
        //RETR_EXTERNAL - retrieves only the extreme outer contours. It sets `hierarchy[i][2]=hierarchy[i][3]=-1` for all the contours.
        //CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points.
        //For example, an up-right rectangular contour is encoded with 4 points.
        
        findContours(imgPre, contours, hierarchy, RetrievalModes::RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        //drawContours(imgTmp, contours, -1, GREEN, 2, LINE_8, hierarchy, 0);

        auto circles = findCircles(contours, hierarchy);
        drawBoundaries(imgTmp, circles);

        //auto [imgColor, mask] = myColorFinder.update(imgTmp, &my_hsv);

        //imshow(windowName + "1", imgTmp);
        //imshow(windowName + "2", imgColor);
        //imshow(windowName + "3", mask);

        //Mat concatImages;// = Mat(img.rows, img.cols, img.type());
        //std::println("img[{0},{1}] imgPre[{2},{3}]", img.cols, img.rows, imgPre.cols, imgPre.rows);
        //hconcat(img, imgPre, concatImages);
    
        imshow(windowName + "2", imgTmp);
        imshow(windowName, imgPre);
        
        //imshow(windowName, concatImages);
    }

    destroyAllWindows();

    return 0;
}