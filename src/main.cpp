#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "my_utilities.h"
#include "cam.h"
#include "camera.h"
#include "picp_solver.h"

int main(int argc, char** argv) {
    int n_meas = 120;   // Total number of measurements (frames)
    int idx1 = 0;       // Starting frame index
    int idx2 = 1;       // Next frame index (increased iteratively)

    // Load and initialize the data
    std::vector<Measurement> measurements = load_and_initialize_data("./data/meas-", n_meas);

    // Camera object
    Cam cam;
    cv::Mat R(3, 3, CV_32F);
    cv::Mat t(3, 1, CV_32F);
    cv::Mat world_points;

    // World map and camera initialization
    pr::Camera pr_cam = pr::Camera(480, 640, cam.getEigenCamera(), Eigen::Isometry3f::Identity());
    pr::Vector3fVector pr_world_points;

    vector<Eigen::Isometry3f> predicted_poses, gt_poses;
    predicted_poses.push_back(Eigen::Isometry3f::Identity());
    gt_poses.push_back(Eigen::Isometry3f::Identity()); // also add following gt poses to vector

    std::vector<std::vector<Eigen::Vector2i>> all_matched_ids_meas;

    // for (int i = 0; i < 3; ++i) {
    for (int i = 0; i < measurements.size(); ++i) {
        idx1 = i;
        idx2 = i + 1;

        std::cout << "Iteration: " << i+1;

        // check for correspondences handling points already known
        std::vector<Eigen::Vector2i> matched_ids_real, matched_ids_meas;
        cv::Mat matched_points1, matched_points2;
        match_points(measurements, idx1, idx2, matched_points1, matched_points2, matched_ids_meas, matched_ids_real);

        // std::cout << ", matched_points1 size: " << matched_points1.size() << ", matched_points2 size: " << matched_points2.size() << std::endl;

        int num_inliers = matched_points1.size[0];
        int total_points = measurements[idx1].points2D.size[0] > measurements[idx2].points2D.size[0] 
                         ? measurements[idx1].points2D.size[0] 
                         : measurements[idx2].points2D.size[0];
        int num_outliers = total_points - num_inliers;
        std::cout << ", number of inliers: " << num_inliers << ", number of outliers: " << num_outliers << std::endl;
        
        all_matched_ids_meas.push_back(matched_ids_meas);

        // printVector2i(matched_ids_meas);
        // check them with gt

        world_points.create(matched_points1.rows, 3, CV_32F); 

        if (i == 0) {
            // First iteration: Triangulate to get the initial world map
            cv::Mat mask;
            cam.computeEssentialAndRecoverPose(matched_points1, matched_points2, R, t, mask);

            world_points.create(matched_points1.rows, 3, CV_32F);
            cam.triangulatePoints(R, t, matched_points1, matched_points2, world_points);
            
            // std::cout << "world_points:\n" << world_points << std::endl;
            
            pr_world_points = matToV3fV_Type21(world_points);
            
            // TODO check for correctness
            Eigen::Isometry3f rel_pose = createIsometryFromRt(cvToEigenMatrix(R), cvToEigenVector(t));
            predicted_poses.push_back(rel_pose.inverse());

            // Initialize camera pose from R, t
            pr_cam.setWorldInCameraPose(rel_pose);

        } else {
            // find matches between previous frame and current
            // Subsequent iterations: Use ICP to refine the camera pose
            pr::Vector2fVector pr_image_points = matToV2fV(measurements[idx2].points2D);

            std::vector<Eigen::Vector2i> common_correspondences;
            std::vector<Eigen::Vector2i> new_correspondences;
            cv::Mat new_world_points;

            filter_correspondences(all_matched_ids_meas, 
                                  common_correspondences,
                                  new_correspondences,
                                  world_points,
                                  new_world_points);

            std::cout << "\nNumber of mutual correspondences between frame " << i << " and frame " << i+1 << ": " << common_correspondences.size() << std::endl;

            pr::IntPairVector correspondences = imgToWorldCorrespondences(world_points, matched_ids_meas);

            // Run ICP to update camera pose
            oneRound(pr_cam, pr_world_points, pr_image_points, correspondences);
        }

        // Print the updated camera pose after each iteration
        // std::cout << "\nUpdated Camera Pose: " << std::endl;
        //printIsometry(pr_cam.worldInCameraPose());

        // Visualize the matches
        // visualizeMatches(matched_points1, matched_points2);
    }
    // printIsometry(pr_cam.worldInCameraPose());

    return 0;
}
