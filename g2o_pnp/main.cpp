#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>

void find_matchs(cv::Mat& img1, cv::Mat& img2,
    std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
    std::vector<cv::DMatch>& nmatches)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(img1, kp1);
    detector->detect(img2, kp2);
    cv::Mat desp1, desp2;
    detector->compute(img1, kp1, desp1);
    detector->compute(img2, kp2, desp2);
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(desp1, desp2, nmatches);
    std::vector<cv::DMatch> good_matches(nmatches.size());
    auto min_max = std::minmax_element(nmatches.begin(), nmatches.end(), [](cv::DMatch match1, cv::DMatch match2){
        return match1.distance < match2.distance;
    });
    double min_dist = min_max.first->distance;
    std::cout << "min_dist: " << min_dist << std::endl;
    auto it = std::copy_if(nmatches.begin(), nmatches.end(), good_matches.begin(), [min_dist](cv::DMatch match){
        if(match.distance <= std::max(2.0 * min_dist, 30.0)){
            return true;
        }
        return false;
    });
    good_matches.resize(std::distance(good_matches.begin(), it));
    std::cout << good_matches.size() << std::endl;
    nmatches = good_matches;
}   

class PnPErrorEdge:public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void computeError() override {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d pose_vu = _K * (v1->estimate() * _pos);
        // 归一化
        pose_vu /=  pose_vu[2];
        _error = _measurement - pose_vu.head(2);
    }
    virtual bool read(std::istream& is) override {return true;}
    virtual bool write(std::ostream& os) const override {return true;}
    void setParams(Eigen::Matrix3d K, Eigen::Vector3d pos){_K = K; _pos = pos;}
private:
    Eigen::Matrix<double, 3, 3> _K;
    Eigen::Vector3d _pos;
};

// 像素坐标转归一化三维坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

int main(int argc, char** argv) {
    if(argc != 4) {
        std::cout << "params are img1 img2 img2_depth" << std::endl;
        return 0;
    }
    std::vector<Eigen::Vector3d> pt_3d;
    std::vector<Eigen::Vector2d> pt_2d;
    cv::Mat img1 = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);
    cv::Mat img1_depth = cv::imread(argv[3], cv::ImreadModes::IMREAD_UNCHANGED);
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch> matches;
    find_matchs(img1, img2, kp1, kp2, matches);
    double fx = 520.9;
    double cx = 325.1;
    double fy = 521.0;
    double cy = 249.7;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Eigen::Matrix3d K1;
    K1 << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;

    // 找到kp1对应的深度信息
    for(const auto& match : matches){
        cv::Point2d pt1 = kp1[match.queryIdx].pt;
        cv::Point2d pt2 = kp2[match.trainIdx].pt;
        double z = img1_depth.at<uint16_t>(int(pt1.y), int(pt1.x)) / 5000.0;
        if(img1_depth.at<uint16_t>(int(pt1.y), int(pt1.x)) == 0)
            continue;
        pt_3d.push_back(Eigen::Vector3d(
            (pt1.x - cx) / fx * z,
            (pt1.y - cy) / fy * z,
            z
        ));
        pt_2d.push_back(Eigen::Vector2d(pt2.x, pt2.y));
    }

    std::cout << "matchs: " << matches.size() << std::endl;
    std::cout << "depth_size: " << pt_3d.size() << std::endl;

    // 用g2o做位姿优化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
    v->setId(0);
    v->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(v);

    
    for(int i=0; i < pt_3d.size(); i++) {
        PnPErrorEdge* e = new PnPErrorEdge();
        e->setId(i);
        e->setParams(K1, pt_3d[i]);
        e->setMeasurement(pt_2d[i]);
        e->setInformation(Eigen::Matrix2d::Identity());
        e->setVertex(0, v);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::cout << "T: " << v->estimate() << std::endl;
}