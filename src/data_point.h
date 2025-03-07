// data_point.h
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
