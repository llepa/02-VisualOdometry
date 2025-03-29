#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "../src/my_utilities.h"
#include "../src/defs.h"
#include "../src/data_point.h"
#include "../src/cam.h"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>

int main(int argc, char** argv) {
    // ---------------------------
    // Initialization
    // ---------------------------
    int n_meas = 121;                                // Total number of measurements (frames)
    std::string meas_path_prefix = "./data/meas-";   // Path prefix for measurement files
    std::string world_filename = "./data/world.dat"; // Ground truth world points file (unused)

    // Load measurements (and optionally ground truth world points)
    std::vector<Measurement> measurements = load_and_initialize_data(meas_path_prefix, n_meas);

    // Set up camera and solvers
    Cam cam;
    pr::Camera picp_cam(480, 640, cam.getEigenCamera(), Eigen::Isometry3f::Identity());
    pr::PICPSolver picp_solver;

    WorldPointVector world_points, triangulated_world_points;
    std::vector<Eigen::Isometry3f> poses, gt_poses;

    // Initialize with identity pose
    poses.push_back(Eigen::Isometry3f::Identity());

    std::cout << "\nIteration: " << "0" << std::endl;

    std::vector<Data_Point> points_first = measurements[0].data_points;
    std::vector<Data_Point> points_second = measurements[1].data_points;
    IntPairVector img_correspondences; // For image point correspondences

    std::vector<std::pair<Data_Point, Data_Point>> initial_matches;
    match_points(points_first, points_second, initial_matches, img_correspondences);

    cv::Mat mask;
    cam.computeEssentialAndRecoverPose(initial_matches, mask);
    Eigen::Isometry3f initial_estimated_pose = cam.getPose();
    std::cout << "Pose: \n" << initial_estimated_pose.matrix() << std::endl;

    cam.triangulatePoints(
        Eigen::Isometry3f::Identity(),
        initial_estimated_pose,
        initial_matches,
        world_points
    );
    // poses.push_back(initial_estimated_pose);

    for (int i = 0; i < n_meas - 1; i++) {
        // Save ground truth pose (after augmentation)
        Eigen::Isometry3f gt_pose = augment_pose(measurements[i].gt_pose);
        gt_poses.push_back(gt_pose);

        std::cout << "\nIteration: " << i << std::endl;

        // Get data points for the current and next measurements
        std::vector<Data_Point> curr_points = measurements[i].data_points;
        std::vector<Data_Point> next_points = measurements[i + 1].data_points;

        // Match current image points with world points
        IntPairVector img_world_correspondences;
        std::vector<std::pair<Data_Point, World_Point>> img_world_matches;
        match_points(next_points, world_points, img_world_matches, img_world_correspondences);

        // Set previous pose and initialize the PICP solver
        Eigen::Isometry3f previous_pose = poses.back();
        picp_cam.setWorldInCameraPose(previous_pose.inverse());

        picp_solver.init(
            picp_cam,
            extract_V3fV(world_points),
            extract_V2fV(next_points)
        );
        picp_solver.setKernelThreshold(3000.0f);

        const int maxIterations = 50;
        float prevError = std::numeric_limits<float>::max();
        float currentError = prevError;
        const float convergenceThreshold = 0.00001f;
        bool convergenceReached = false;

        for (int j = 0; j < maxIterations; j++) {
            if (!picp_solver.oneRound(img_world_correspondences, false)) {
                std::cerr << "Solver iteration " << j << " failed." << std::endl;
                break;
            }
            currentError = picp_solver.chiInliers();
            float relImprovement = (prevError > 1e-10) ? std::abs(prevError - currentError) / prevError : 0.0f;
            if (relImprovement < convergenceThreshold) {
                convergenceReached = true;
                std::cout << "Convergence reached at iteration " << j << std::endl;
                break;
            }
            prevError = currentError;
        }
        if (!convergenceReached) {
            std::cerr << "Convergence not reached." << std::endl;
        }
        std::cout << "PICP inliers: " << picp_solver.numInliers() << "/" << img_world_correspondences.size() << std::endl;

        // Recover the estimated pose from the solver
        Eigen::Isometry3f estimated_pose = picp_solver.camera().worldInCameraPose().inverse();
        std::cout << "Estimated pose\n" << estimated_pose.matrix() << std::endl;
        std::cout << "Gt pose\n" << gt_pose.matrix() << std::endl;
        poses.push_back(estimated_pose);

        // Find additional image point matches for triangulation
        IntPairVector img_correspondences_local;
        std::vector<std::pair<Data_Point, Data_Point>> img_matches;
        match_points(curr_points, next_points, img_matches, img_correspondences_local);

        std::vector<std::pair<Data_Point, Data_Point>> new_points_to_triangulate =
            add_new_world_points(img_world_matches, img_matches);

        cam.triangulatePoints(
            previous_pose,
            estimated_pose,
            new_points_to_triangulate,
            world_points
        );

        triangulated_world_points.clear();
        std::cout << "Number of world points: " << world_points.size() << std::endl;
    }

    Eigen::Isometry3f gt_pose = augment_pose(measurements[n_meas - 1].gt_pose);
    gt_poses.push_back(gt_pose);

    for (size_t j = 0; j < poses.size(); j++) {
        poses[j] = cam.cameraToImage() * poses[j];
    }

    Eigen::Affine3f alignmentTransform = alignTrajectories(poses, gt_poses);

    std::ofstream outputEstimatedTrajectory("output/estimated_trajectory.txt");
    std::ofstream outputEstimatedTrajectoryScaled("output/estimated_trajectory_scaled.txt");
    std::ofstream outputErrors("output/errors.txt");
    std::ofstream outputEstimatedWorldPoints("output/estimated_world_points.txt");

    if (!outputEstimatedTrajectory.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!outputEstimatedTrajectoryScaled.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!outputErrors.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!outputEstimatedWorldPoints.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        exit(EXIT_FAILURE);
    }

    float scale = alignmentTransform.linear().col(0).norm();


    int counter = 0;
    for (size_t j = 0; j < poses.size(); j++) {
        Eigen::Isometry3f gt_pose = gt_poses[j];
        Eigen::Isometry3f& pose = poses[j];

        float angle_gt = std::atan2(gt_pose.rotation()(1, 0), gt_pose.rotation()(0, 0));
        float angle = std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
        float pi = 3.1415926535897932384626433832795028841971693;
        angle += pi / 2.0;


        outputEstimatedTrajectory << counter << " " << pose.translation().x() << " "
            << pose.translation().y() << " " << angle << "\n";

        pose.translation() = pose.translation() * scale;
        outputEstimatedTrajectoryScaled << counter << " " << pose.translation().x() << " "
           << pose.translation().y() << " " << angle << "\n";

        float error_translation = (pose.translation() - gt_pose.translation()).norm();
        float error_rotation = std::abs(angle - angle_gt);

        outputErrors << counter << " " << error_translation
                                << " " << error_rotation << "\n";

        counter++;
    }

    for (int id = 0; id < 1000; id++) {
        for (auto& wp : world_points) {
            if (wp.id_real == id) {
                Eigen::Vector3f wp_eigen = Eigen::Vector3f(wp.coordinates.x, wp.coordinates.y, wp.coordinates.z);
                Eigen::Vector3f transformed_wp = cam.cameraToImage() * wp_eigen * scale;
                outputEstimatedWorldPoints << id << " " << transformed_wp.x()
                                                 << " " << transformed_wp.y()
                                                 << " " << transformed_wp.z() << "\n";
                break;
            }
        }
    }

    create_plot(gt_poses, poses, "Trajectory");

    return 0;
}
