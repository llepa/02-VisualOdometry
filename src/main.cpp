#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

// 3. Include your project-specific headers
#include "my_utilities.h"
#include "camera.h"
#include "picp_solver.h"
#include "defs.h"
#include "data_point.h"

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
    cv::Mat world_points_gt = load_world_points(world_filename);

    // Initialize camera objects
    Cam cam; 
    pr::Camera pr_cam(480, 640, cam.getEigenCamera(), Eigen::Isometry3f::Identity());

    // Initialize world_points as an empty Nx3 matrix
    cv::Mat world_points = cv::Mat::zeros(0, 3, CV_32F);

    // Initialize pose arrays and error accumulators
    std::vector<Eigen::Isometry3f> estimated_poses, gt_poses;
    std::vector<float> errors_rotation, errors_translation;

    // Set initial poses (camera and ground truth) to identity
    estimated_poses.emplace_back(Eigen::Isometry3f::Identity());
    gt_poses.emplace_back(Eigen::Isometry3f::Identity());

    // This will store all the correspondences between frames
    std::vector<std::vector<Eigen::Vector2i>> all_correspondences;

    // Process each pair of consecutive frames
    for (int i = 0; i < n_meas - 1; ++i) {
        int idx1 = i;
        int idx2 = i + 1;

        std::cout << "\nIteration: " << i + 1 << std::endl;

        // Match points between frame idx1 and idx2
        std::vector<Eigen::Vector2i> correspondences_meas, correspondences_gt;
        std::vector<std::pair<Data_Point, Data_Point>> matches;
        std::vector<Data_Point> points1 = measurements[idx1].data_points;
        std::vector<Data_Point> points2 = measurements[idx2].data_points;

        // extract points from mesurements
        match_points(
            points1,
            points2,
            matches,
            correspondences_meas,
            correspondences_gt
            );

        int num_inliers = matches.size();
        int total_points = std::max((int)measurements[idx1].data_points.size(), 
                                    (int)measurements[idx2].data_points.size());
        int num_outliers = total_points - num_inliers;
        std::cout << "Number of inliers: " << num_inliers << ", Number of outliers: " << num_outliers << std::endl;

        // Store the correspondences for later use in filtering
        all_correspondences.push_back(correspondences_meas);

        // Augment ground truth pose for the current frame i
        Eigen::Isometry3f gt_pose = augment_pose(measurements[idx1].gt_pose);
        gt_poses.push_back(gt_pose);

        if (i == 0) {
            // First iteration: Compute essential matrix and recover initial relative pose
            cv::Mat mask;
            cam.computeEssentialAndRecoverPose(points1, points2, mask);
            
            cv::Mat R = cam.getRotationMatrix();
            cv::Mat t = cam.getTranslationVector();

            cv::Mat T1 = cv::Mat::eye(4, 4, CV_32F);
            cv::Mat T2 = cv::Mat::eye(4, 4, CV_32F);
            R.copyTo(T2(cv::Range(0,3), cv::Range(0,3)));
            t.copyTo(T2(cv::Range(0,3), cv::Range(3,4)));

            // Triangulate initial set of 3D points
            cam.triangulatePoints(T1, T2, points1, points2, world_points);

            // Convert rotation and translation to Eigen
            Eigen::Matrix3f R_eigen = cvToEigenMatrix(R);
            Eigen::Vector3f t_eigen = cvToEigenVector(t);
            Eigen::Isometry3f rel_pose = createIsometryFromRt(R_eigen, t_eigen); // This is world_to_camera

            pr_cam.setWorldInCameraPose(rel_pose);
            estimated_poses.push_back(rel_pose.inverse()); // Store as camera_to_world

        } else {
            // Subsequent iterations
            std::vector<Eigen::Vector2i> common_correspondences;
            std::vector<Eigen::Vector2i> new_correspondences;
            cv::Mat common_world_points;
            // cv::Mat filtered_img_points1, filtered_img_points2;
            std::vector<Data_Point> filtered_img_points1, filtered_img_points2;

            // Convert measurement points of next frame to Eigen vectors for PICP
            const pr::Vector2fVector pr_image_points = data_point_v_to_v2fv(measurements[idx2].data_points);

            std::cout << "*** Filtering correspondences..." << std::endl;

            // Filter correspondences to separate common and new matches, and update world points if needed
            filter_correspondences(all_correspondences, 
                                   common_correspondences,
                                   new_correspondences,
                                   world_points,
                                   common_world_points);

            // Update world_points if common_world_points is not empty
            if (!common_world_points.empty()) {
                world_points = common_world_points.clone();
            }

            // Get last estimated pose (camera_to_world)
            Eigen::Isometry3f last_pose_estimate = estimated_poses.back();
            pr::Vector3fVector pr_world_points = matToV3fV(world_points);

            std::cout << "*** Computing one round..." << std::endl;
            
            // Convert common correspondences to PICP format
            pr::IntPairVector correspondences_world = eigenToIntPairVector(common_correspondences);

            // Estimate new pose using one round of PICP
            Eigen::Isometry3f estimated_pose = oneRound(last_pose_estimate.inverse(), // world_to_camera
                                                        pr_cam, 
                                                        pr_world_points, 
                                                        pr_image_points, 
                                                        correspondences_world); // returns world_to_camera
            
            // Store the new estimated pose as camera_to_world
            estimated_poses.push_back(estimated_pose.inverse());

            // Prepare transformations for triangulation
            Eigen::Isometry3f T1 = last_pose_estimate.inverse(); // world_to_camera of previous pose
            Eigen::Isometry3f T2 = estimated_pose.inverse();     // world_to_camera of current pose

            std::cout << "Filtering matches..." << std::endl;

            // Filter new correspondences to get matched image points
            filter_matches(measurements[idx1].data_points, 
                           measurements[idx2].data_points, 
                           new_correspondences, 
                           filtered_img_points1, 
                           filtered_img_points2);

            std::cout << "*** Triangulating points..." << std::endl;

            // Triangulate new 3D points using updated poses
            cv::Mat new_world_points_local;

            cam.triangulatePoints(
                isometry3fToCvMat(T1),
                isometry3fToCvMat(T2),
                filtered_img_points1,
                filtered_img_points2,
                new_world_points_local
                );

            std::cout << "Previous world points: " << world_points.rows << std::endl;
            std::cout << "Newly found world points: " << new_world_points_local.rows << std::endl;

            // Append new world points if found
            if (!new_world_points_local.empty()) {
                cv::vconcat(world_points, new_world_points_local, world_points);
            }

            std::cout << "World points after adding new found points: " << world_points.rows << std::endl;
        }

        // Compute and accumulate errors after each iteration
        if (estimated_poses.size() > 1) {
            // Relative pose between last two estimated poses
            size_t last_idx = estimated_poses.size() - 1;
            Eigen::Isometry3f relative_pose = estimated_poses[last_idx - 1].inverse() * estimated_poses[last_idx];
            Eigen::Isometry3f relative_pose_gt = gt_poses[last_idx - 1].inverse() * gt_poses[last_idx];

            // Rotation error
            Eigen::Matrix3f R_err = relative_pose.linear().transpose() * relative_pose_gt.linear();
            float angle_error = std::acos((R_err.trace() - 1.0f) / 2.0f) * (180.0f / M_PI);
            errors_rotation.push_back(angle_error);

            // Translation error
            Eigen::Vector2d translation_computed(relative_pose.translation()(0), relative_pose.translation()(1));
            Eigen::Vector2d translation_gt(relative_pose_gt.translation()(0), relative_pose_gt.translation()(1));
            float translation_error = (translation_computed - translation_gt).norm();
            errors_translation.push_back(translation_error);

            std::cout << "Rotational error: " << angle_error << " degrees" << std::endl;
            std::cout << "Translational error: " << translation_error << " units" << std::endl;
        }
    }

    // After processing all frames, output cumulative error statistics
    std::cout << "\nRotational errors (per-frame): ";
    for (const auto& err : errors_rotation) 
        std::cout << err << " ";
    std::cout << "\n" << std::endl;

    std::cout << "Translational errors (per-frame): ";
    for (const auto& err : errors_translation) 
        std::cout << err << " ";
    std::cout << std::endl;
    std::cout << "\n" << std::endl;

    // Plot the estimated trajectory against ground truth
    create_plot(gt_poses, estimated_poses, "Trajectory Plot");

    return 0;
}
