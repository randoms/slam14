#include <iostream>
#include <fstream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <chrono>


int main (int argc, char** argv) {

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    std::ifstream g2o_file("./sphere.g2o");
    while (!g2o_file.eof())
    {
        std::string name;
        g2o_file >> name;
        if(name == "VERTEX_SE3:QUAT") {
            g2o::VertexSE3* v = new g2o::VertexSE3();
            int index = 0;
            g2o_file >> index;
            v->setId(index);
            v->read(g2o_file);
            optimizer.addVertex(v);
            if(index == 0)
                v->setFixed(true);
        }
        if(name == "EDGE_SE3:QUAT") {
            static int edge_count = 0;
            g2o::EdgeSE3* e = new g2o::EdgeSE3();
            int index1, index2;
            g2o_file >> index1 >> index2;
            e->setId(edge_count);
            // std::cout << "index1: " << index1 << std::endl;
            // std::cout << "index2: " << index2 << std::endl;
            // std::cout << "optimizer.vertices()[index1]: " << optimizer.vertices()[index1] << std::endl;
            // std::cout << "optimizer.vertices()[index2]: " << optimizer.vertices()[index2] << std::endl;
            e->setVertex(0, optimizer.vertices()[index1]);
            e->setVertex(1, optimizer.vertices()[index2]);
            e->read(g2o_file);
            optimizer.addEdge(e);
            edge_count += 1;
        }
        if(!g2o_file.good()) break;
    }

    std::cout << "g2o vertex: " << optimizer.vertices().size() << std::endl;
    std::cout << "g2o edges: " << optimizer.edges().size() << std::endl;
    
    optimizer.initializeOptimization();
    optimizer.optimize(40);
    optimizer.save("test.g2o");
    optimizer.clear();
    std::cout << "edges: " << optimizer.edges().size() << std::endl;
    std::cout << "vertex: " << optimizer.vertices().size() << std::endl;
    optimizer.load("sphere.g2o");
    std::cout << "loaded" << std::endl;
    std::cout << "edges: " << optimizer.edges().size() << std::endl;
    std::cout << "vertex: " << optimizer.vertices().size() << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(40);
    optimizer.save("test1.g2o");
    // set to bal problem
    
    return 0;
}