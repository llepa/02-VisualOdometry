#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "my_utilities.cpp"

int main() {

    int n_meas = 120;   
    vector<Measurement> measurements = extract_measurements("./data/meas-", n_meas);
    //Camera camera = extract_camera("./data/camera.dat");
    Camera camera = Camera();
    vector<Point3D> points3D = extract_world("./data/world.dat");

    // Match the descriptors of the two images using a Brute-Force matcher
    cv::Mat matched_ids = cv::Mat_<int>(0, 2);

    // Declare two matches matrices
    cv::Mat matched_points1 = cv::Mat_<double>(0, 2);
    cv::Mat matched_points2 = cv::Mat_<double>(0, 2);

    match_points(measurements, 0, 1, matched_points1, matched_points2, matched_ids);

    cv::Mat K = camera.K;
    K.convertTo(K, 6);
    cv::Mat E = cv::findEssentialMat(matched_points1, matched_points2, K); 
    
    // recoverPose
    cv::Mat R, t, mask, points4D_tr;
    // int inliers = cv::recoverPose(E, matched_points1, matched_points2, K, R, t);
    int inliers = recoverPose(E, matched_points1, matched_points2, K, R, t, 100.0, mask, points4D_tr);
    cout << endl;
    cout << "R: " << endl << R << endl;
    cout << "t: " << endl << t << endl;
    cout << endl;

    // print inliers
    cout << "inliers: " << endl << inliers << endl;

    cv::Mat points3D_tr;
    cv::convertPointsFromHomogeneous(points4D_tr.t(), points3D_tr);

    // print points3D_tr
    cout << "points3D_tr: " << endl << points3D_tr << endl;
    cout << endl;

    // triangulate
    // cv::Mat points3D_tr = triangulate(R, t, K, matched_points1, matched_points2);

    visualize_coordinates(points3D_tr, matched_points1, matched_points2);
    
    /*
    // work on ground truth data
    */
    
    return 0;
}