#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cv::Mat img_1 = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat desp1, desp2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> desptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    desptor->compute(img_1, keypoints_1, desp1);
    desptor->compute(img_2, keypoints_2, desp2);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = t2 - t1;
    std::cout << "extract ORB cost: " << duration.count() << std::endl;

    cv::Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB extracted", outimg1);

    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(desp1, desp2, matches);
    t2 = std::chrono::steady_clock::now();
    duration = t2 -t1;
    std::cout << "match ORB cost: " << duration.count() << std::endl;

    auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2){
        return m1.distance < m2.distance;
    });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::cout << "min distance: " << min_dist << std::endl;
    std::cout << "max distance: " << max_dist << std::endl;

    std::vector<cv::DMatch> good_matches(matches.size());

    auto it = std::copy_if(matches.begin(), matches.end(), good_matches.begin(), [min_dist](cv::DMatch match){

        if(match.distance <= std::max(2.00 * min_dist, 30.0))
            return true;
        return false;
    });

    good_matches.resize(std::distance(good_matches.begin(), it));

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_1, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("matchs", img_match);
    cv::imshow("good matches", img_goodmatch);
    std::cout << "good matches: " << good_matches.size() << std::endl;
    cv::waitKey(0);

    return 0;

}