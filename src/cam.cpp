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
    z_near = 0.001; // Prevent points at zero depth
    z_far = 5;
    width = 640;
    height = 480; 
}

void Cam::computeEssentialAndRecoverPose(const std::vector<std::pair<Data_Point, Data_Point>> &matches,
                                         cv::Mat& mask) {

    // Set fixed random seed for reproducible results
    cv::setRNGSeed(42);
    
    std::vector<cv::Point2f> cv_points1, cv_points2;
    extract_coordinates_from_matches(matches, cv_points1, cv_points2);

    // Configure RANSAC parameters for better robustness
    double ransacThreshold = 1.0; // In pixels
    double ransacConfidence = 0.999; // High confidence
    int maxIters = 2000; // More iterations for better results
    
    cv::Mat E = cv::findEssentialMat(
        cv_points1,
        cv_points2,
        K_cv,
        cv::RANSAC,
        ransacConfidence,
        ransacThreshold,
        mask
        );

    if (E.empty()) {
        std::cerr << "Essential matrix computation failed!" << std::endl;
        return;
    }

    // Count inliers
    int inliers = cv::countNonZero(mask);
    std::cout << "Essential matrix estimation: " << inliers << " inliers out of " 
              << matches.size() << " matches (" 
              << (100.0f * inliers / matches.size()) << "%)" << std::endl;

    // Recover pose with the same parameters for consistency
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

    // Compute projection matrices using the intrinsic matrix K_cv
    cv::Mat P1 = K_cv * T1(cv::Range(0, 3), cv::Range(0, 4));
    cv::Mat P2 = K_cv * T2(cv::Range(0, 3), cv::Range(0, 4));

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1, points2, points4D);

    // Convert from homogeneous coordinates
    points4D = points4D.t();
    cv::Mat points3D_temp(points4D.rows, 3, CV_32F);
    std::vector<bool> valid_points(points4D.rows, false);
    int valid_count = 0;

    // First pass: Convert to 3D and check point validity
    for (int i = 0; i < points4D.rows; ++i) {
        float w = points4D.at<float>(i, 3);
        if (std::fabs(w) > std::numeric_limits<float>::epsilon()) {
            float x = points4D.at<float>(i, 0) / w;
            float y = points4D.at<float>(i, 1) / w;
            float z = points4D.at<float>(i, 2) / w;
            
            // Check if the point is in front of both cameras
            cv::Mat pt1 = T1(cv::Range(0, 3), cv::Range(0, 3)) * cv::Mat(cv::Vec3f(x, y, z)) + 
                          T1(cv::Range(0, 3), cv::Range(3, 4));
            cv::Mat pt2 = T2(cv::Range(0, 3), cv::Range(0, 3)) * cv::Mat(cv::Vec3f(x, y, z)) + 
                          T2(cv::Range(0, 3), cv::Range(3, 4));
            
            if (pt1.at<float>(2) > z_near && pt2.at<float>(2) > z_near && 
                z > z_near && z < z_far) {
                points3D_temp.at<float>(i, 0) = x;
                points3D_temp.at<float>(i, 1) = y;
                points3D_temp.at<float>(i, 2) = z;
                valid_points[i] = true;
                valid_count++;
            } else {
                // Point is behind camera or too far/close
                points3D_temp.at<float>(i, 0) = 0.0f;
                points3D_temp.at<float>(i, 1) = 0.0f;
                points3D_temp.at<float>(i, 2) = 0.0f;
            }
        } else {
            // Point at infinity
            points3D_temp.at<float>(i, 0) = 0.0f;
            points3D_temp.at<float>(i, 1) = 0.0f;
            points3D_temp.at<float>(i, 2) = 0.0f;
        }
    }

    // Create final output with only valid points
    if (valid_count > 0) {
        cv::Mat filtered_points(valid_count, 3, CV_32F);
        int idx = 0;
        for (int i = 0; i < points4D.rows; ++i) {
            if (valid_points[i]) {
                points3D_temp.row(i).copyTo(filtered_points.row(idx++));
            }
        }
        points3D = filtered_points.clone();
        
        std::cout << "Triangulated " << valid_count << " valid points out of " 
                  << points4D.rows << " matches" << std::endl;
    } else {
        std::cout << "No valid points triangulated." << std::endl;
        points3D = cv::Mat();
    }
}


void Cam::_computeProjectionMatrices(const cv::Mat& R1, 
                                     const cv::Mat& t1,
                                     const cv::Mat& R2, 
                                     const cv::Mat& t2,
                                     cv::Mat& P1, 
                                     cv::Mat& P2) {
    // Create [R|t] matrices
    cv::Mat R_t1 = cv::Mat::zeros(3, 4, CV_32F);
    R1.copyTo(R_t1(cv::Rect(0, 0, 3, 3)));
    t1.copyTo(R_t1(cv::Rect(3, 0, 1, 3)));

    cv::Mat R_t2 = cv::Mat::zeros(3, 4, CV_32F);
    R2.copyTo(R_t2(cv::Rect(0, 0, 3, 3)));
    t2.copyTo(R_t2(cv::Rect(3, 0, 1, 3)));

    // Compute projection matrices
    P1 = K_cv * R_t1;
    P2 = K_cv * R_t2;
}


void Cam::visualizePoints(const cv::Mat& point_matrix) {
    if (point_matrix.empty()) {
        std::cout << "No points to visualize." << std::endl;
        return;
    }

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

// This function is no longer needed as we handle scaling differently
void Cam::normalize_translation(cv::Mat& t) {
    // Intentionally left empty - we don't want to normalize the translation
    // as it would distort the scale of the reconstruction
}
