#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);

    if (image.data == nullptr)
    {
        std::cerr << "read image failed: " << argv[1] << std::endl;
        return 0;
    }

    std::cout << "image width: " << image.cols << std::endl;
    std::cout << "image height: " << image.rows << std::endl;

    cv::imshow("image", image);
    cv::waitKey(0);

    if (image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        std::cout << "please select an gray scale image" << std::endl;
        return 0;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; y++)
    {
        uint8_t *row_ptr = image.ptr<uint8_t>(y);
        for (size_t x = 0; x < image.cols; x++)
        {
            uint8_t *data_ptr = &row_ptr[image.channels() * x];
            for (int c = 0; c < image.channels(); c++)
            {
                uint8_t data = data_ptr[c];
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "lit image used: " << time_used.count() << std::endl;

    cv::Mat image_another = image;
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::imshow("image", image_another);
    cv::waitKey(0);

    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;

}