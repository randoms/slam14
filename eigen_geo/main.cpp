#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv) {

    Eigen::Matrix3d rotate_matrix = Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0,0,1));
    std::cout.precision(3);
    std::cout << "rotation matrix = " << rotate_matrix.matrix() << std::endl;
    rotate_matrix = rotation_vector.toRotationMatrix();
    std::cout << "rotation matrix = " << rotate_matrix.matrix() << std::endl;
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1, 0, 0) after rotation (by angle axis) = " << v_rotated.transpose() << std::endl;
    v_rotated = rotate_matrix * v;
    std::cout << "(1, 0, 0) after rotation (by rotation matrix) " << v_rotated.transpose() << std::endl;

    Eigen::Vector3d eular_angle = rotate_matrix.eulerAngles(2,1,0);
    std::cout << "yaw pitch roll: " << eular_angle.transpose() << std::endl;

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,3,4));
    std::cout << "Transform matrix: " << T.matrix() << std::endl;
    
    Eigen::Vector3d v_transformed = T * v;
    std::cout << "v transformed: " << v_transformed.matrix() << std::endl;

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    std::cout << "quaternion from rotation vector: " << q.coeffs().transpose() << std::endl;
    q = Eigen::Quaterniond(rotate_matrix);
    std::cout << "quaternion from rotation matrix: " << q.coeffs().transpose() << std::endl;
    v_rotated = q * v;
    std::cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << std::endl;
    std::cout << "should be equal to " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl;




    return 0;
}