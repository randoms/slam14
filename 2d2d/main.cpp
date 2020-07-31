#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

void find_feature_matches(const cv::Mat& img1, const cv::Mat& img2,
    std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& matches
);

void pose_estimation_2d2d(
    std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& nmatches,
    cv::Mat& R, cv::Mat& t
);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void find_feature_matches(const cv::Mat& img1, const cv::Mat& img2,
    std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& matches
){
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Mat dp1, dp2;
    detector->detect(img1, kp1);
    detector->detect(img2, kp2);
    detector->compute(img1, kp1, dp1);
    detector->compute(img2, kp2, dp2);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::BRUTEFORCE_HAMMING
    );
    matcher->match(dp1, dp2, matches);
    // 进行错误过滤
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](cv::DMatch match1, cv::DMatch match2){
            return match1.distance < match2.distance;
        }
    );
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::vector<cv::DMatch> good_matches(matches.size());
    auto it = std::copy_if(matches.begin(), matches.end(), good_matches.begin(), [min_dist](cv::DMatch match) {
        if(match.distance <= std::max(2.0 * min_dist, 30.0))
            return true;
        return false;
    });
    good_matches.resize(std::distance(good_matches.begin(), it));
    matches = good_matches;
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K){
    return cv::Point2d (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(
    std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& nmatches,
    cv::Mat& R, cv::Mat& t
){
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point2d> pt1;
    std::vector<cv::Point2d> pt2;

    for(int i = 0; i < nmatches.size(); i ++) {
        pt1.push_back(kp1[nmatches[i].queryIdx].pt);
        pt2.push_back(kp2[nmatches[i].trainIdx].pt);
    }

    cv::Mat f_mat = cv::findFundamentalMat(pt1, pt2, cv::FM_8POINT);
    std::cout << "fundamental matrix: " << std::endl << f_mat << std::endl;

    cv::Point2d p_point(325.1, 249.7);
    double f = 521;
    cv::Mat e_mat = cv::findEssentialMat(pt1, pt2, f, p_point);
    std::cout << "e matrix: " << std::endl << e_mat << std::endl;

    cv::Mat h_mat = cv::findHomography(pt1, pt2, cv::FM_RANSAC, 3);
    std::cout << "h matrix: " << std::endl << h_mat << std::endl;

    cv::recoverPose(e_mat, pt1, pt2, R, t, f, p_point);

    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t << std::endl;
}


int main(int argc, char** argv) {
    cv::Mat img1 = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch> nmatches;
    find_feature_matches(img1, img2, kp1, kp2, nmatches);
    
    std::cout << "find matches: " << nmatches.size() << std::endl;
    cv::Mat R, t;
    pose_estimation_2d2d(kp1, kp2, nmatches, R, t);
    cv::Mat t_x =
    (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cv::Mat e_mat = t_x * R;
    std::cout << "e mat from R, t: " << std::endl << e_mat << std::endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(const auto& match:nmatches){
        cv::Point2d pt1 = kp1[match.queryIdx].pt;
        pt1 = pixel2cam(pt1, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = kp2[match.trainIdx].pt;
        pt2 = pixel2cam(pt2, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "d: " << d << std::endl;
    }

    return 0;
}
