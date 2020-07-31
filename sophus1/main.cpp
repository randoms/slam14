#include <iostream>
#include <fstream>

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

TrajectoryType readTraj(const std::string &filepath)
{
    std::ifstream fin(filepath);
    if(!fin){
        std::cerr << "traj file: " << filepath << " not found" << std::endl;
    }

    TrajectoryType res;

    while (!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d SE3_Rt(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz));
        res.push_back(SE3_Rt);
    }
    return res;
}


int main(int argc, char** argv) {
    // 从实际位姿文件读取位姿
    std::string groundtruth_file = "../groundtruth.txt";
    std::string estimated_file = "../estimated.txt";
    TrajectoryType SE3_truth = readTraj(groundtruth_file);
    TrajectoryType SE3_estimate = readTraj(estimated_file);
    // 转到se3表示
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> Trajse3;
    // 计算误差
    double sum = 0;
    for(int i=0; i< SE3_truth.size(); i ++) {
        sum += (SE3_estimate[i].inverse() * SE3_truth[i]).log().squaredNorm();
    }
    std::cout.precision(3);
    std::cout << "Error is " << sqrt(sum / SE3_truth.size()) << std::endl;
    return 0;
}