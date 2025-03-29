#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

// 3. Include your project-specific headers
#include "../src/my_utilities.h"
#include "../src/camera.h"
#include "../src/picp_solver.h"
#include "../src/defs.h"
#include "../src/data_point.h"
#include "../src/cam.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// main function
int main(int argc, char** argv) {
    // Configuration parameters
    int n_meas = 120;   // Total number of measurements (frames)
    std::string meas_path_prefix = "./data/meas-";    // Path prefix for measurement files
    std::string world_filename = "./data/world.dat";  // Ground truth world points file

    // Load measurements and ground truth world points
    std::vector<Measurement> measurements = load_and_initialize_data(meas_path_prefix, n_meas);
    std::vector<World_Point> world_points_gt = load_world_points(world_filename);

    // Initialize camera objects
    Cam cam = Cam();
    // Camera pr_cam(
    //     cam.getHeight(),
    //     cam.getWidth(),
    //     cam.getEigenCamera(),
    //     Eigen::Isometry3f::Identity()
    // );

    // Initialize world_points as an empty matrix (no rows, 3 columns)
    // cv::Mat world_points = cv::Mat();
    std::vector<World_Point> world_points, triangulated_world_points;

    // Initialize pose arrays and error accumulators
    std::vector<Eigen::Isometry3f> estimated_poses, gt_poses;
    std::vector<float> errors_rotation, errors_translation;
    std::vector<float> scale_factors;

    // Set initial poses (camera and ground truth) to identity
    estimated_poses.push_back(Eigen::Isometry3f::Identity());
    gt_poses.push_back(Eigen::Isometry3f::Identity());

    // pr::Camera picp_cam(cam.getHeight(), cam.getWidth(), cam.getEigenCamera(), Eigen::Isometry3f::Identity());
    // pr::PICPSolver picp_solver;

    // Process each pair of consecutive frames
    for (int i = 0; i < n_meas - 1; ++i) {
        int idx1 = i;
        int idx2 = i + 1;

        std::cout << "\nIteration: " << i + 1 << std::endl;

        // Match points between frame idx1 and idx2
        IntPairVector img_correspondences, img_correspondences_gt;
        IntPairVector img_world_correspondences;
        std::vector<Data_Point> points1 = measurements[idx1].data_points;
        std::vector<Data_Point> points2 = measurements[idx2].data_points;

        if (i == 0) {

            Eigen::Isometry3f gt_pose = augment_pose(measurements[idx1].gt_pose);
            gt_poses.push_back(gt_pose);

            std::vector<std::pair<Data_Point, Data_Point>> matches;

            match_points(
                points1,
                points2,
                matches,
                img_correspondences
            );

            int num_inliers = matches.size();
            int total_points = std::max(
                static_cast<int>(measurements[idx1].data_points.size()),
                static_cast<int>(measurements[idx2].data_points.size())
            );
            int num_outliers = total_points - num_inliers;
            std::cout << "Number of inliers: " << num_inliers << ", Number of outliers: " << num_outliers << std::endl;

            cv::Mat mask;
            cam.computeEssentialAndRecoverPose(matches, mask);

            cam.initOneRound(world_points, points2);

            Eigen::Isometry3f estimated_pose = cam.getPose();

            cam.triangulatePoints(
                Eigen::Isometry3f::Identity(),
                estimated_pose,
                matches,
                world_points
            );

            // cam.projectPoints(world_points);

            std::cout << "Number of world points: " << world_points.size() << std::endl;

            // cam.setPose(T_eigen);
            // Eigen::Isometry3f estimated_pose = cam.getPose();
            estimated_poses.push_back(estimated_pose); // Store as camera_to_world
            std::cout << "Estimated pose: translation: " << estimated_pose.translation() << std::endl;
            std::cout << "Estimated pose: rotation:    " << estimated_pose.rotation().eulerAngles(0, 1, 2) << std::endl;

        } else {

            Eigen::Isometry3f gt_pose = augment_pose(measurements[idx1].gt_pose);
            gt_poses.push_back(gt_pose);

            std::vector<std::pair<Data_Point, World_Point>> img_world_matches;

            match_points(
                points2,
                world_points,
                img_world_matches,
                img_world_correspondences
            );

            Eigen::Isometry3f previous_pose = estimated_poses.back();

            cam.setPose(previous_pose);
            cam.initOneRound(world_points, points2);
            cam.oneRound(img_world_correspondences);

            Eigen::Isometry3f estimated_pose = cam.getPose();

            estimated_poses.push_back(estimated_pose);

            // std::cout << "Pose: \n" << estimated_pose.matrix() << std::endl;

            std::vector<std::pair<Data_Point, Data_Point>> img_matches;

            match_points(
                points1,
                points2,
                img_matches,
                img_correspondences
            );

            std::vector<std::pair<Data_Point, Data_Point>> points_to_triangulate =
                add_new_world_points(
                    img_world_matches,
                    img_matches
                );


            cam.triangulatePoints(
                previous_pose,
                estimated_pose,
                points_to_triangulate,
                world_points
            );

            check_world_points_sanity(world_points);

            triangulated_world_points.clear();

            std::cout << "Number of world points: " << world_points.size() << std::endl;

            // std::cout << "Pose: translation: estimatated: "
            //           << "x: " << estimated_pose.translation().coeff(0)
            //           << ", y: " << estimated_pose.translation().coeff(1) << "\n"
            //           << "Pose: translation: gt:          "
            //           << "x: " << gt_pose.translation().coeff(0)
            //           << ", y: " << gt_pose.translation().coeff(1) << std::endl;
            //
            // Eigen::Quaternionf q(estimated_pose.rotation().matrix());
            // float yaw = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), 1 - 2*(q.y()*q.y() + q.z()*q.z()));
            // std::cout << "Pose: rotation: estimated / gt: "
            //           << yaw
            //           << " / "
            //           << gt_pose.rotation().eulerAngles(0, 1, 2).coeff(2) << std::endl;
        }

        // Compute and accumulate errors after each iteration
        // if (estimated_poses.size() > 1) {
        //     size_t last_idx = estimated_poses.size() - 1;
        //     Eigen::Isometry3f pose = estimated_poses.back();
        //     Eigen::Isometry3f pose_gt = gt_poses.back();
        //
        //     pose.translation().z() = 0.0f;
        //
        //     // Compute translation norms.
        //     float trans_estimated = pose.translation().norm();
        //     float trans_gt = pose_gt.translation().norm();
        //
        //     // Only compute scale factor if both translations are significant.
        //     if (trans_gt > 0.001f && trans_estimated > 1e-6f) {
        //         float current_scale = trans_gt / trans_estimated;
        //         scale_factors.push_back(current_scale);
        //         std::cout << "Scale factor: " << current_scale << std::endl;
        //     }
        //
        //     // Accumulate translational error.
        //     float translation_error = std::abs(trans_gt - trans_estimated);
        //     errors_translation.push_back(translation_error);
        //     std::cout << "Translation error: " << translation_error << std::endl;
        //
        //     Eigen::AngleAxisf angleAxis_est(pose.rotation());
        //     Eigen::AngleAxisf angleAxis_gt(pose_gt.rotation());
        //     float angle_diff = std::abs(angleAxis_est.angle() - angleAxis_gt.angle());
        //     errors_rotation.push_back(angle_diff);
        //     std::cout << "Angular difference (radians): " << angle_diff << std::endl;
        // }

    }

    float total_gt_distance = 0.0f, total_est_distance = 0.0f;
    size_t N = std::min(gt_poses.size(), estimated_poses.size());
    for (size_t i = 1; i < N; ++i) {
        total_gt_distance += (gt_poses[i].translation() - gt_poses[i - 1].translation()).norm();
        total_est_distance += (estimated_poses[i].translation() - estimated_poses[i - 1].translation()).norm();
    }
    float absolute_scale = (total_est_distance > 1e-6f) ? (total_gt_distance / total_est_distance) : 1.0f;
    std::cout << "\nAbsolute scale factor: " << absolute_scale << std::endl;

    // --- Apply absolute scaling to all estimated poses except the first one (identity) ---
    std::vector<Eigen::Isometry3f> scaled_poses;
    scaled_poses.push_back(estimated_poses[0]);

    for (size_t i = 1; i < estimated_poses.size(); ++i) {
        Eigen::Isometry3f scaled_pose = cam.cameraToImage() * estimated_poses[i];
        scaled_poses.push_back(scaled_pose);
    }


    // Output cumulative error statistics
    std::cout << "\nRotational errors (per-frame): ";
    for (const auto& err : errors_rotation)
        std::cout << err << " ";
    std::cout << "\n" << std::endl;

    std::cout << "Translational errors (per-frame): ";
    for (const auto& err : errors_translation)
        std::cout << err << " ";
    std::cout << std::endl;
    std::cout << "\n" << std::endl;

    // Plot the scaled estimated trajectory against ground truth
    create_plot(gt_poses, scaled_poses, "Trajectory");

    return 0;
}
