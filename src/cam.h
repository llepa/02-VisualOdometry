#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

class Cam {

public:
    Cam();

    void computeEssentialAndRecoverPose(const cv::Mat& matched_points1, 
                                        const cv::Mat& matched_points2,
                                        cv::Mat& R, 
                                        cv::Mat& t, 
                                        cv::Mat& mask);

    
    void triangulatePoints(const cv::Mat& R, 
                           const cv::Mat& t,
                           const cv::Mat& matched_points1,
                           const cv::Mat& matched_points2,
                           cv::Mat& points3D);


    cv::Mat projectPoints(const cv::Mat& R, 
                          const cv::Mat& t, 
                          const cv::Mat& points3D);


    void visualizePoints(const cv::Mat& point_matrix);


    Eigen::Matrix3f getEigenCamera();

private:
    cv::Mat K_cv;
    Eigen::Matrix3f K_eig;
    float z_near;
    float z_far;
    int width;
    int height;

    void _computeProjectionMatrices(const cv::Mat& R, 
                                    const cv::Mat& t,
                                    cv::Mat& P1, 
                                    cv::Mat& P2);

};

