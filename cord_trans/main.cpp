#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv) {
    // 1 号 位姿 q = [0.35, 0.2, 0.3, 0.1], t = [0.3, 0.1, 0.1]
    // 2 号 位姿 q = [-0.5, 0.4, -0.1, 0.2], t = [-0.1, 0.5, 0.3]

    Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    q1.normalize();
    Eigen::Vector3d t1(0.3, 0.1, 0.1);

    Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    q2.normalize();
    Eigen::Vector3d t2(-0.1, 0.5, 0.3);

    Eigen::Isometry3d T1(q1);
    T1.pretranslate(t1);

    Eigen::Isometry3d T2(q2);
    T2.pretranslate(t2);

    Eigen::Vector3d v1(0.5, 0, 0.2);

    Eigen::Vector3d v2 = T2 * T1.inverse() * v1;

    std::cout << "v in robot2: " << v2.transpose() << std::endl;

    return 0;
}