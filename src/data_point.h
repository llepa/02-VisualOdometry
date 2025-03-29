#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>

struct Data_Point {
    int id_meas;
    int id_real;
    cv::Point2f coordinates;
    Eigen::VectorXf descriptor;

    Data_Point(int meas_id, int real_id, cv::Point2f coord, const Eigen::VectorXf& desc)
        : id_meas(meas_id), id_real(real_id), coordinates(coord), descriptor(desc) {}

    Data_Point()
        : id_meas(0), id_real(0), coordinates(0.0f, 0.0f), descriptor(Eigen::VectorXf()) {}
};

struct World_Point {
    cv::Point3f coordinates;
    Eigen::VectorXf descriptor;
    int id_real;
    int id_meas;

    World_Point(cv::Point3f coord, const Eigen::VectorXf& desc,  int id_real)
        : coordinates(coord), descriptor(desc), id_real(id_real), id_meas(-1) {}

    World_Point(cv::Point3f coord, const Eigen::VectorXf& desc,  int id_meas, int id_real)
        : coordinates(coord), descriptor(desc), id_meas(id_meas), id_real(id_real) {}

};

typedef std::vector<Data_Point> DataPointVector;
typedef std::vector<World_Point> WorldPointVector;
// typedef std::vector<IntPair> IntPairVector;
