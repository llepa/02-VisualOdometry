#include "cam.h"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

Cam::Cam() {
    K_cv = (cv::Mat_<float>(3, 3) << 180, 0, 320,
                                     0, 180, 240,
                                     0,   0,   1);
    K_eig = (Eigen::Matrix3f() << 180, 0, 320, 
                                  0, 180, 240, 
                                  0,   0,   1).finished();
    z_near = 0.001; // original value was 0
    z_far = 5;
    width = 640;
    height = 480; 
}

void Cam::computeEssentialAndRecoverPose(const cv::Mat& matched_points1, 
                                         const cv::Mat& matched_points2,
                                         cv::Mat& R, 
                                         cv::Mat& t, 
                                         cv::Mat& mask) {
    // float focal = K_cv.at<float>(0, 0);
    // cv::Point2d pp(K_cv.at<float>(0, 2), K_cv.at<float>(1, 2));
    // cv::Mat E = findEssentialMat(matched_points1, matched_points2, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    // recoverPose(E, matched_points1, matched_points2, R, t, focal, pp, mask);

    cv::Mat E = findEssentialMat(matched_points1, matched_points2, K_cv, cv::RANSAC);
    cv::recoverPose(E, matched_points1, matched_points2, K_cv, R, t, mask);
}

void Cam::triangulatePoints(const cv::Mat& R, 
                            const cv::Mat& t,
                            const cv::Mat& matched_points1,
                            const cv::Mat& matched_points2,
                            cv::Mat& points3D) {

    // std::cout << "Triangulating points..." << std::endl;
    cv::Mat P1, P2;
    _computeProjectionMatrices(R, t, P1, P2);
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, matched_points1.t(), matched_points2.t(), points4D);

    cv::convertPointsFromHomogeneous(points4D.t(), points3D);

    // for (int i = 0; i < points4D.cols; i++) {
    //     float w = points4D.at<float>(3, i);
    //     points3D.at<float>(i, 0) = points4D.at<float>(0, i) / w;
    //     points3D.at<float>(i, 1) = points4D.at<float>(1, i) / w;
    //     points3D.at<float>(i, 2) = points4D.at<float>(2, i) / w;
    // }

    // std::cout << "Triangulation finished" << std::endl;
}

void Cam::_computeProjectionMatrices(const cv::Mat& R, 
                                    const cv::Mat& t,
                                    cv::Mat& P1, 
                                    cv::Mat& P2) {
    P1 = K_cv * cv::Mat::eye(3, 4, CV_32F);
    cv::Mat R_t = cv::Mat::zeros(3, 4, CV_32F);

    // std::cout << "R: " << "\n" << R << std::endl;
    // std::cout << "t: " << "\n" << t << std::endl;

    R.copyTo(R_t(cv::Rect(0, 0, 3, 3)));
    t.copyTo(R_t(cv::Rect(3, 0, 1, 3)));

    // std::cout << "R_t: " << "\n" << R_t << std::endl;
    
    P2 = K_cv * R_t;
}

cv::Mat Cam::projectPoints(const cv::Mat& R, const cv::Mat& t, const cv::Mat& points3D) {
    cv::Mat projected_points(points3D.rows, 2, CV_32F);
    cv::Mat P1, P2;
    _computeProjectionMatrices(R, t, P1, P2);
    for (int i = 0; i < points3D.rows; i++) {
        cv::Mat point3D = (cv::Mat_<float>(4, 1) << points3D.at<float>(i, 0), 
                          points3D.at<float>(i, 1), points3D.at<float>(i, 2), 1.0);
        cv::Mat img_point = P2 * point3D;
        float x = img_point.at<float>(0, 0) / img_point.at<float>(2, 0);
        float y = img_point.at<float>(1, 0) / img_point.at<float>(2, 0);
        projected_points.at<float>(i, 0) = x;
        projected_points.at<float>(i, 1) = y;
    }

    return projected_points;
}

void Cam::visualizePoints(const cv::Mat& point_matrix) {
    // Create a window
    cv::viz::Viz3d window("3D Points");

    // Create a vector to hold the 3D points
    std::vector<cv::Point3f> cloud;

    // Fill the cloud vector with points from the input matrix
    for (int i = 0; i < point_matrix.rows; i++) {
        cv::Point3f p(point_matrix.at<float>(i, 0),
                      point_matrix.at<float>(i, 1),
                      point_matrix.at<float>(i, 2));
        cloud.push_back(p);
    }

    // Create a WCloud object and set its properties
    cv::viz::WCloud cloud_widget(cloud, cv::viz::Color::white());
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 5);

    // Add the WCloud object to the window
    window.showWidget("cloud", cloud_widget);

    // Create a coordinate system
    cv::viz::WCoordinateSystem cs(1.0);
    window.showWidget("CoordinateSystem", cs);

    // Show the visualization
    window.spin();
}


Eigen::Matrix3f Cam::getEigenCamera() {
    return K_eig;
}

