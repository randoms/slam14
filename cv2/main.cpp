#include <opencv2/opencv.hpp>
#include <string>

std::string image_file = "./distroted.png";

int main(int argc, char **argv)
{
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    cv::Mat image = cv::imread(image_file);
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);
    for (int v = 0; v < rows; v++){
        for (int u = 0; u < cols; u++){
            double x = (u -cx) / fx, y = (u - cy) / fy;
            double r = sqrt(x*x + y*y);
            
        }
    }
}