#include "cam.h"

#include "my_utilities.h"
#include "picp_solver.h"
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

    cameraToImageTransform = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f m = Eigen::Matrix3f::Zero();
    m(0, 2) =  1;
    m(1, 0) = -1;
    m(2, 1) = -1;
    // m(0, 1) = -1;
    // m(1, 2) = -1;
    // m(2, 0) =  1;
    cameraToImageTransform.linear() = m;

    z_near = 0; // Prevents points a zero-depth
    z_far = 5;
    width = 640;
    height = 480;

    picp_cam = Camera(height, width, K_eig, Eigen::Isometry3f::Identity());
    picp_solver = PICPSolver();
}

void Cam::computeEssentialAndRecoverPose(const std::vector<std::pair<Data_Point, Data_Point>> &matches,
                                         cv::Mat& mask) {

    cv::setRNGSeed(42);

    std::vector<cv::Point2f> cv_points1, cv_points2;
    extract_coordinates_from_matches(matches, cv_points1, cv_points2);

    double ransacThreshold = 1.0; // In pixels
    double ransacConfidence = 0.999; // High confidence
    int maxIters = 2000; // More iterations for better

    cv::Mat E = findEssentialMat(
        cv_points1,
        cv_points2,
        K_cv,
        cv::RANSAC
        );

    if (E.empty()) {
        std::cerr << "Essential matrix computation failed!" << std::endl;
        exit(EXIT_FAILURE);
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

    Eigen::Matrix3f R_eigen;
    Eigen::Vector3f t_eigen;
    Eigen::Isometry3f T_eigen;

    cv2eigen(R_, R_eigen);
    cv2eigen(t_, t_eigen);

    T_eigen.linear() = R_eigen;
    T_eigen.translation() = t_eigen;

    picp_cam.setWorldInCameraPose(T_eigen.inverse());

    // return T_eigen.inverse();

    // std::cout << "Mask: /n" << mask << std::endl;

    // t_ = -t_;   // ???

    // std::cout << "Rotation matrix (R):\n" << R_ << std::endl;
    // std::cout << "Translation vector (t):\n" << t_ << std::endl;
}


void Cam::triangulatePoints(const Eigen::Isometry3f& T1_eigen,
                            const Eigen::Isometry3f& T2_eigen,
                            std::vector<std::pair<Data_Point, Data_Point>>& matches,
                            std::vector<World_Point>& points3D) {

    cv::Mat new_world_points;
    std::vector<cv::Point2f> points1, points2;
    extract_coordinates_from_matches(matches, points1, points2);

    if (points1.empty() || points2.empty()) {
        std::cout << "Skipping triangulation: not enough points." << std::endl;
        return;
    }

    cv::Mat T1, T2;
    cv::eigen2cv(T1_eigen.inverse().matrix(), T1);
    cv::eigen2cv(T2_eigen.inverse().matrix(), T2);
    cv::Mat P1 = K_cv * T1(cv::Range(0, 3), cv::Range(0, 4));
    cv::Mat P2 = K_cv * T2(cv::Range(0, 3), cv::Range(0, 4));

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1, points2, points4D);
    points4D = points4D.t();

    convertPointsFromHomogeneous(points4D, new_world_points);

    std::cout << "Number of triangulated world points before checking duplicates: " << new_world_points.rows << std::endl;

    for (size_t i = 0; i < new_world_points.rows; i++) {
        cv::Point3f pt;
        pt.x = new_world_points.at<float>(i, 0);
        pt.y = new_world_points.at<float>(i, 1);
        pt.z = new_world_points.at<float>(i, 2);

        // Eigen::Vector3f pt_img_eig(pt.x, pt.y, pt.z);

        // Eigen::Vector3f pt_camera_eig = cameraToImageTransform * pt_img_eig;
        // cv::Point3f pt_image_cv(pt_camera.x(), pt_camera.y(), pt_camera.z());

        Eigen::VectorXf desc = matches[i].first.descriptor;
        int id_meas = matches[i].first.id_meas;
        int id_real = matches[i].first.id_real;
        World_Point wp(pt, desc, id_meas, id_real);

        points3D.push_back(wp);
    }
}


Eigen::Matrix3f Cam::getEigenCamera() {
    return K_eig;
}


int Cam::getHeight() const {
    return height;
}


int Cam::getWidth() const {
    return width;
}


void Cam::projectPoints(std::vector<World_Point>& world_points) {
    cv::Mat rvec;
    cv::Rodrigues(R_, rvec);

    // Project the 3D points to 2D using the pose (rvec, t) and camera matrix K
    std::vector<cv::Point2f> projectedPoints;

    cv::projectPoints(
        extract_matrix_coordinates(world_points),
        rvec,
        t_,
        K_cv,
        cv::Mat(),
        projectedPoints
    );

    // std::cout << "Projected points: \n" << projectPoints << std::endl;
}


void Cam::initOneRound(std::vector<World_Point> world_points, std::vector<Data_Point> img_points) {

    _world_points_picp = extract_V3fV(world_points);
    _image_points_picp = extract_V2fV(img_points);
    picp_solver.init(picp_cam, _world_points_picp, _image_points_picp);
    picp_solver.setKernelThreshold(1000.0f);

    if (!picp_cam.worldInCameraPose().isApprox(picp_solver.camera().worldInCameraPose())) {
        std::cerr << "Cam::initOneRound failed: camera poses are different!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Cam::oneRound(IntPairVector correspondences) {
    const int maxIterations = 50;
    double prevError = std::numeric_limits<double>::max();
    double currentError = prevError;
    const double convergenceThreshold = 0.01;

    // for (int i = 0; i < maxIterations; i++) {
    //     if (!picp_solver.oneRound(correspondences, false)) {
    //         std::cerr << "Solver iteration " << i << " failed." << std::endl;
    //         break;
    //     }
    //
    //     currentError = picp_solver.chiInliers();
    //     double relImprovement = (prevError > 1e-10) ? std::abs(prevError - currentError) / prevError : 0.0;
    //
    //     if (relImprovement < convergenceThreshold) {
    //         std::cout << "Convergence reached at iteration " << i << std::endl;
    //         break;
    //     }
    //     prevError = currentError;
    //
    // }

    for (int i = 0; i < 5; i++) {
        picp_solver.oneRound(correspondences, false);
    }

    picp_cam = picp_solver.camera();
    std::cout << "PICP inliers: " << picp_solver.numInliers() << "/" << correspondences.size() << std::endl;
    if (!picp_cam.worldInCameraPose().isApprox(picp_solver.camera().worldInCameraPose())) {
        std::cerr << "Cam::oneRound failed: camera poses are different!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Eigen::Isometry3f Cam::getPose() {
    return picp_cam.worldInCameraPose();
}

void Cam::setPose(Eigen::Isometry3f pose) {
    picp_cam.setWorldInCameraPose(pose);
}

Eigen::Isometry3f Cam::cameraToImage() {
    return cameraToImageTransform;
}


