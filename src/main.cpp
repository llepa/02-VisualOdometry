#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "my_utilities.cpp"

int main() {

    // Load and initialize data
    int n_meas = 120;   
    vector<Measurement> measurements = extract_measurements("../data/meas-", n_meas);
    Camera camera = Camera(); // Default camera
    vector<Point3D> points3D = extract_world("../data/world.dat");

    // Prepare matrices for matching
    cv::Mat matched_ids = cv::Mat_<int>(0, 2);
    cv::Mat matched_points1 = cv::Mat_<double>(0, 2);
    cv::Mat matched_points2 = cv::Mat_<double>(0, 2);

    // Match the points from two measurements
    match_points(measurements, 0, 1, matched_points1, matched_points2, matched_ids);

    // Extract intrinsic camera matrix and set data type
    cv::Mat K = camera.K;
    K.convertTo(K, 6);

    // Compute Essential matrix and recover pose
    cv::Mat R, t, mask;
    double focal = camera.K.at<double>(0, 0); // Focal length
    cv::Point2d pp(camera.K.at<double>(0, 2), camera.K.at<double>(1, 2));  // Principal point

    cv::Mat E = findEssentialMat(matched_points2, matched_points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    recoverPose(E, matched_points2, matched_points1, R, t, focal, pp, mask);

    // Compute the projection matrices for the two views
    cv::Mat P1 = camera.K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat R_t = cv::Mat::zeros(3, 4, CV_64F); 
    R.copyTo(R_t(cv::Rect(0, 0, 3, 3)));
    t.copyTo(R_t(cv::Rect(3, 0, 1, 3)));
    cv::Mat P2 = camera.K * R_t;

    // Triangulate points to get the 3D world coordinates
    cv::Mat points4D(4, matched_points1.cols, CV_64F);
    cv::triangulatePoints(P1, P2, matched_points1.t(), matched_points2.t(), points4D);

    // Convert homogeneous 4D points to 3D
    cv::Mat points3D_tr(points4D.cols, 3, CV_64F);
    for (int i = 0; i < points4D.cols; i++) {
        double w = points4D.at<double>(3, i);  // Homogeneous coordinate
        
        points3D_tr.at<double>(i, 0) = points4D.at<double>(0, i) / w;
        points3D_tr.at<double>(i, 1) = points4D.at<double>(1, i) / w;
        points3D_tr.at<double>(i, 2) = points4D.at<double>(2, i) / w;
    }

    // Visualize the 3D points
    visualize_3d_points(points3D_tr);

    /*
    // TODO: Work on ground truth data
    */
    
    return 0;
}
