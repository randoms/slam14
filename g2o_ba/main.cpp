#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/core/robust_kernel_impl.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include<sophus/so3.hpp>

#include "common.h"

class ProjectErrorEdge: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSBAPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ProjectErrorEdge(Eigen::Matrix3d K, double k1, double k2):_K(K), _k1(k1), _k2(k2){
        //  resize(3);
    }
    virtual void computeError() override {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        // const g2o::VertexSBAPointXYZ* v3 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);

        Eigen::Vector3d pt = _K * (v1->estimate() * v2->estimate());
        pt /= (-pt[2]);
        double r2 = pt.squaredNorm();
        double params = (1 + _k1 * r2 + _k2 * r2 * r2);
        pt[0] = pt[0] * params;
        pt[1] = pt[1] * params;
        _error = _measurement - pt.head(2);
    }
    virtual bool read(std::istream &is) override {return true;}
    virtual bool write(std::ostream &os) const override {return true;}
private:
    Eigen::Matrix3d _K;
    double _k1;
    double _k2;
};

struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = Sophus::SO3d::exp(Eigen::Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        q = rotation.unit_quaternion();
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
        K << focal, 0, 0, 0, focal, 0, 0, 0, 1;
    }

    /// 将估计值放入内存
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Quaterniond q;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
};

int main (int argc, char** argv) {

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    BALProblem bal_problem("./problem-16-22106-pre.txt");
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("./init.ply");
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    std::vector<g2o::VertexSE3Expmap *> camera_vertex;
    std::vector<g2o::VertexSBAPointXYZ *> point_vertex;
    std::vector<PoseAndIntrinsics> pose_ints;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        auto pose = PoseAndIntrinsics(camera);
        pose_ints.push_back(pose);
        v->setEstimate(g2o::SE3Quat(pose.q, pose.translation));
        optimizer.addVertex(v);
        camera_vertex.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Eigen::Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        point_vertex.push_back(v);
    }

    const double *observations = bal_problem.observations();
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        auto pose = pose_ints[bal_problem.camera_index()[i]];
        ProjectErrorEdge *edge = new ProjectErrorEdge(pose.K, pose.k1, pose.k2);
        edge->setVertex(0, camera_vertex[bal_problem.camera_index()[i]]);
        edge->setVertex(1, point_vertex[bal_problem.point_index()[i]]);
        std::cout << "_measurement" << Eigen::Vector2d(observations[2 * i + 0], observations[2 * i + 1]).transpose() << std::endl;
        edge->setMeasurement(Eigen::Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = camera_vertex[i];
        auto estimate = vertex->estimate();
        pose_ints[i].set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = point_vertex[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }

    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}