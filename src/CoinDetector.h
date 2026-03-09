#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <numbers>
#include <tuple>


using namespace cv;
using Contours = std::vector<std::vector<Point>>;
using Hierarchy = std::vector<Vec4i>;
const Scalar GREEN = Scalar(0, 255, 0);


class CoinDetector {
public:
    CoinDetector(Mat image, std::string windowName);
    ~CoinDetector() {};

    std::tuple<Mat, Mat> Update();
    int moneyCount = 0;

private:
    Mat preProcessing(Mat img);
    Contours findCircles(Contours contours, Hierarchy hierarchy);
    std::string classifyCoinBySize(double area_px);
    void labelCoins(Mat img, Contours circles);
    static void on_trackbar(int pos, void *) { return; }

private:
    float _1mmInPixels = 0;
    std::string trackbarCanny1 = "CannyThr1";
    std::string trackbarCanny2 = "CannyThr2";
    std::string trackbarScale = "Scale";
    std::string _windowName = "TrackBars";

    Mat color_img;
    Mat gray_img;

    std::unordered_map<std::string, float> coinDiametersInMM = {
        {"1gr", 15.50},
        {"10gr", 16.50},
        {"2gr", 17.50},
        {"20gr", 18.50},
        {"5gr", 19.50},
        {"50gr", 20.50},

        {"2zl", 21.50},
        {"1zl", 23.00},
        {"5zl", 24.00}
    };

    std::unordered_map<std::string, float> coinValues = {
        {"1gr", 1},
        {"2gr", 2},
        {"5gr", 5},
        {"10gr", 10},
        {"20gr", 20},
        {"50gr", 50},
        {"1zl", 100},
        {"2zl", 200},
        {"5zl", 500}
    };

};


CoinDetector::CoinDetector(Mat image, std::string windowName = "") {
    color_img = image.clone();
    cvtColor(image, gray_img, COLOR_BGR2GRAY);

    if (windowName != "") _windowName = windowName;

    namedWindow(_windowName);
    createTrackbar(trackbarCanny1, _windowName, nullptr, 255, on_trackbar);
    setTrackbarPos(trackbarCanny1, _windowName, 255);

    createTrackbar(trackbarCanny2, _windowName, nullptr, 255, on_trackbar);
    setTrackbarPos(trackbarCanny2, _windowName, 7);

    createTrackbar(trackbarScale, _windowName, nullptr, 1000, on_trackbar);
    setTrackbarPos(trackbarScale, _windowName, 304);
}


Mat CoinDetector::preProcessing(Mat img) {

    Mat blur_img;
    int n = 2;
    GaussianBlur(img, blur_img, Size(2*n+1, 2*n+1), 0);

    Mat canny_img;
    int thresh1 = getTrackbarPos(trackbarCanny1, _windowName);
    int thresh2 = getTrackbarPos(trackbarCanny2, _windowName);
    Canny(blur_img, canny_img, thresh1, thresh2);

    Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(3, 3));

    Mat dialated_img;
    //int morph_oper1 = getTrackbarPos(trackbarMorph1, _windowName);
    morphologyEx(canny_img, dialated_img, MorphTypes::MORPH_DILATE, kernel);

    Mat morph_close_img;
    //int morph_oper2 = getTrackbarPos(trackbarMorph2, _windowName);
    morphologyEx(dialated_img, morph_close_img, MorphTypes::MORPH_CLOSE, kernel);

    return morph_close_img;
}

Contours CoinDetector::findCircles(Contours contours, Hierarchy hierarchy) {
    Contours circles;

    float epsilon = 0.15;
    //foundRectangle = false;

    for (const auto &cnt : contours) {

        double peri = arcLength(cnt, true);
        std::vector<Point> approxPolyVec;
        approxPolyDP(cnt, approxPolyVec, 0.02 * peri, true);

        auto polyNum = approxPolyVec.size();
        //std::println("Polygons: [{0}]", polyNum);

        auto rect = cv::boundingRect(cnt);

        ////niestety wyniki nie s¹ zbyt dok³adne
        //if (polyNum == 4) {
        //    foundRectangle = true;
        //    //std::println("{0}", foundRectangle);
        //    _1mmInPixels = ( rect.height - (rect.height * 0.1) ) / 10.0;
        //    std::println("{0}", _1mmInPixels);
        //}

        if (polyNum > 5) {
            float ratio = rect.height / static_cast<float>(rect.width);
            ratio = std::min(ratio, 1.0f / ratio);

            if (ratio >= 1.0f - epsilon) {
                circles.emplace_back(cnt);
            }
        }

    }

    return circles;
}

std::string CoinDetector::classifyCoinBySize(double area_px) {

    std::string closest_coin = "";
    float min_error = std::numeric_limits<float>::max();

    float _1mmInPixelsTmp = 0;
    /*if (foundRectangle) {
        _1mmInPixelsTmp = _1mmInPixels;
    }*/
    //else {

    _1mmInPixelsTmp = getTrackbarPos(trackbarScale, _windowName) / 100.0; // /100 konwertuje dane z trackbara na poprawne jednostki
    //}

    double pi = std::numbers::pi;

    for (const auto &[coin_name, diameter_mm] : coinDiametersInMM) {
        float r1 = _1mmInPixelsTmp * diameter_mm * 0.5;
        float expected_area_px = pi * r1 * r1;
        float error = std::abs(area_px - expected_area_px);

        if (error < min_error) {
            closest_coin = coin_name;
            min_error = error;
        }
    }
    return closest_coin;
}


void CoinDetector::labelCoins(Mat img, Contours circles) {
    int total_count = 0;

    for (const auto &circle : circles) {
        auto rect = cv::boundingRect(circle);
        rectangle(img, rect, GREEN, 2);

        Point center = Point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);

        cv::circle(img, center, static_cast<int>(rect.width / 2.0), GREEN, 2);
        cv::circle(img, center, 3, GREEN, FILLED);

        auto area = contourArea(circle);
        std::string coin_name = classifyCoinBySize(area);

        int coin_value = coinValues[coin_name];
        total_count += coin_value;

        putText(img, coin_name, rect.tl(), FONT_ITALIC, 0.5, Scalar(255, 0, 0), 3);
        putText(img, coin_name, rect.tl(), FONT_ITALIC, 0.5, Scalar(255, 255, 255), 2);

    }
    std::string amount = "Total: " + std::to_string(total_count / 100) + "zl " + std::to_string(total_count % 100) + "gr";

    Point org = Point(img.cols / 3, img.rows - 10);
    putText(img, amount, org, FONT_ITALIC, 0.5, Scalar(255, 0, 0), 3);
    putText(img, amount, org, FONT_ITALIC, 0.5, Scalar(255, 255, 255), 2);

}


std::tuple<Mat, Mat> CoinDetector::Update() {

    Mat imgColor = color_img.clone();
    Mat imgPre = preProcessing(gray_img);

    Contours contours;
    Hierarchy hierarchy;

    findContours(imgPre, contours, hierarchy, RetrievalModes::RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat imgContours = Mat::ones(imgPre.size(), imgPre.type());
    drawContours(imgContours, contours, -1, Scalar(255, 255, 255), 2, LINE_8, hierarchy, 0);

    auto circles = findCircles(contours, hierarchy);
    labelCoins(imgColor, circles);

    return { imgColor, imgContours };

}