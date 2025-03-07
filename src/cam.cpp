#include "cam.h"

#include "my_utilities.h"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/core/eigen.hpp>


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

void Cam::computeEssentialAndRecoverPose(const std::vector<std::pair<Data_Point, Data_Point>> &matches,
                                         cv::Mat& mask) {

    cv::setRNGSeed(1);
    std::vector<cv::Point2f> cv_points1, cv_points2;
    extract_coordinates_from_matches(matches, cv_points1, cv_points2);

    cv::Mat E = cv::findEssentialMat(
        cv_points1,
        cv_points2,
        K_cv,
        cv::RANSAC
        );

    if (E.empty()) {
        std::cerr << "Essential matrix computation failed!" << std::endl;
        return;
    }

    recoverPose(
        E,
        cv_points1,
        cv_points2,
        K_cv,
        R_,
        t_,
        mask
        );

    std::cout << "Rotation matrix (R):\n" << R_ << std::endl;
    std::cout << "Translation vector (t):\n" << t_ << std::endl;
}


void Cam::triangulatePoints(const cv::Mat& T1,
                            const cv::Mat& T2,
                            std::vector<std::pair<Data_Point, Data_Point>>& matches,
                            cv::Mat& points3D) {

    std::vector<cv::Point2f> points1, points2;
    extract_coordinates_from_matches(matches, points1, points2);

    // Check if we have enough points
    if (points1.empty() || points2.empty()) {
        std::cout << "Skipping triangulation: not enough points." << std::endl;
        return;
    }

    // cv::Mat pts1 = extract_matrix_coordinates(points1);
    // cv::Mat pts2 = extract_matrix_coordinates(points2);

    // Compute projection matrices using the intrinsic matrix K_cv
    cv::Mat P1, P2, points4D;
    P1 = K_cv * T1(cv::Range(0, 3), cv::Range(0, 4));
    P2 = K_cv * T2(cv::Range(0, 3), cv::Range(0, 4));

    std::cout << "T1: \n" << T1 << std::endl;
    std::cout << "T2: \n" << T2 << std::endl;

    // Note: cv::triangulatePoints expects input points as a 2 x N matrix, so we transpose pts1 and pts2.
    cv::triangulatePoints(
        P1,
        P2,
        points1,
        points2,
        points4D);

    // Convert from homogeneous coordinates.
    points4D = points4D.t();
    cv::Mat points3D_temp(points4D.rows, 3, CV_32F);

    for (int i = 0; i < points4D.rows; ++i) {
        float w = points4D.at<float>(i, 3);
        if (std::fabs(w) > std::numeric_limits<float>::epsilon()) {
            points3D_temp.at<float>(i, 0) = points4D.at<float>(i, 0) / w;
            points3D_temp.at<float>(i, 1) = points4D.at<float>(i, 1) / w;
            points3D_temp.at<float>(i, 2) = points4D.at<float>(i, 2) / w;
        } else {
            // Handle the case where w is zero (point at infinity)
            points3D_temp.at<float>(i, 0) = 0.0f;
            points3D_temp.at<float>(i, 1) = 0.0f;
            points3D_temp.at<float>(i, 2) = 0.0f;
        }
    }

    points3D = points3D_temp.clone();
}


void Cam::_computeProjectionMatrices(const cv::Mat& R1, 
                                     const cv::Mat& t1,
                                     const cv::Mat& R2, 
                                     const cv::Mat& t2,
                                     cv::Mat& P1, 
                                     cv::Mat& P2) {

    

    cv::Mat R_t1 = cv::Mat::zeros(3, 4, CV_32F);
    R1.copyTo(R_t1(cv::Rect(0, 0, 3, 3)));
    t1.copyTo(R_t1(cv::Rect(3, 0, 1, 3)));

    cv::Mat R_t2 = cv::Mat::zeros(3, 4, CV_32F);
    R2.copyTo(R_t2(cv::Rect(0, 0, 3, 3)));
    t2.copyTo(R_t2(cv::Rect(3, 0, 1, 3)));

    P1 = K_cv * R_t1;
    P2 = K_cv * R_t2;
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

void Cam::normalize_translation(cv::Mat& t) {
    // Remove or correct this function
    // Incorrectly modifying the translation vector can distort the pose estimation
}
