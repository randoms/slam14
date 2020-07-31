#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>



class ORBErrorEdge:public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ORBErrorEdge(cv::Mat img, cv::KeyPoint kp, cv::Mat desp, Eigen::Matrix3d     K):
        _img(img),_kp(kp), _desp(desp), _K(K)
    {
        _detector = cv::ORB::create();
        _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }

    virtual void computeError() override {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        std::cout  << "_kp"
        Eigen::Vector3d pt1 = _K * (v1->estimate() * Eigen::Vector3d(_kp.pt.x, _kp.pt.y, 1));
        pt1 /= pt1[2];
        // 计算角度变换
        std::cout << "original angle: " << _kp.angle << std::endl;
        Eigen::Vector3d origin_angle(std::cos(_kp.angle / 180), std::sin(_kp.angle / 180), 1);
        std::cout << "v1: " << v1->estimate() << std::endl;
        Eigen::Vector3d after_angle = _K * (v1->estimate().inverse() * (_K.inverse() * origin_angle));
        std::cout << "after_angle: " << after_angle.transpose() << std::endl;
        after_angle /= after_angle[2];
        double angle_degree = std::atan2(after_angle[1], after_angle[0]) + M_PI;
        angle_degree = angle_degree / M_PI * 180;
        std::cout << "after angle degree: " << angle_degree << std::endl;
        std::cout << "pt: " << pt1 << std::endl;
        cv::KeyPoint kp(cv::Point2d(pt1[0], pt1[1]), _kp.size, angle_degree);
        std::cout << "OK1 " << std::endl;
        cv::Mat desp;
        std::vector<cv::KeyPoint> kps;
        std::cout << "OK2 " << std::endl;
        kps.push_back(kp);
        _detector->compute(_img, kps, desp);
        std::cout << "OK3 " << std::endl;
        std::vector<cv::DMatch> matches;
        std::cout << "desp: " << desp << std::endl;
        _matcher->match(desp, _desp, matches);
        std::cout << "distance: " << matches[0].distance << std::endl;          
        _error(0, 0) = _measurement - matches[0].distance;
    }

    virtual bool read(std::istream &in) override{return true;       }
    virtual bool write(std::ostream &out) const override{return true;}
private:
    cv::Mat _img;
    cv::KeyPoint _kp;
    cv::Mat _desp;
    Eigen::Matrix3d _K;
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorMatcher> _matcher;

};

int main(int argc, char** argv) {
    if(argc != 3){
        std::cout << "params: img1 img2" << std::endl;
        return 0;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::ImreadModes::IMREAD_GRAYSCALE);

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> kps;
    cv::Mat desp;
    detector->detect(img1, kps, desp);
    detector->compute(img1, kps, desp);
    std::cout << "find points: " << kps.size() << std::endl;

    // std::vector<cv::KeyPoint> test_kps;

    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
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

    std::cout << "desp col: " << desp.cols << std::endl;
    std::cout << "desp row: " << desp.rows << std::endl;
    for(int i =0; i< kps.size(); i++) {
        ORBErrorEdge* e = new ORBErrorEdge(img2, kps[i], desp.row(i), K);
        e->setId(i);
        e->setMeasurement(0);
        e->setVertex(0, v);
        e->setInformation(Eigen::Matrix<double, 1, 1>(1));
        optimizer.addEdge(e);
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    std::cout << "T: " << v->estimate() << std::endl;

    return 0;
}