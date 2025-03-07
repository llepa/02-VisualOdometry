// my_utilities.h
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <sstream>      // For std::stringstream
#include <iomanip>      // For std::setfill and std::setw

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/core/eigen.hpp> // For cv::eigen2cv and cv::cv2eigen

#include "data_point.h"
#include "cam.h"
#include "picp_solver.h"

struct Measurement {
    int seq;
    Eigen::Vector3f gt_pose;             // Ground truth pose
    Eigen::Vector3f odometry_pose;       // Odometry pose
    std::vector<Data_Point> data_points; // Vector of Data_Point instances

    // Constructor
    Measurement() 
        : seq(0), 
          gt_pose(Eigen::Vector3f::Zero()), 
          odometry_pose(Eigen::Vector3f::Zero()),
          data_points() {}
};

struct World_Point {
    cv::Point3f coordinates;
    Eigen::VectorXf descriptor;
    int id_real;
};

// Type Aliases for Better Readability
using Vector2fVector = std::vector<Eigen::Vector2f>;
using Vector3fVector = std::vector<Eigen::Vector3f>;
using IntPairVector = std::vector<Eigen::Vector2i>;

// Function Declarations

// Utility Functions
float euclidean_distance(const Eigen::VectorXf& vec1, const Eigen::VectorXf& vec2);
std::string size(const cv::Mat& m);
std::vector<cv::Point2f> extract_coordinates(const std::vector<Data_Point>& points);
void extract_coordinates_from_matches(
    std::vector<std::pair<Data_Point, Data_Point>> matches,
    std::vector<cv::Point2f>& matches1,
    std::vector<cv::Point2f>& matches2
);
cv::Mat extract_matrix_coordinates(const std::vector<Data_Point>& points);
void match_points(const std::vector<Data_Point>& data_points1,
                 const std::vector<Data_Point>& data_points2,
                 std::vector<std::pair<Data_Point, Data_Point>>& matches,
                 std::vector<Eigen::Vector2i>& matched_ids_meas,
                 std::vector<Eigen::Vector2i>& matched_ids_real);
std::vector<std::string> split(const std::string& str, const std::string& delimiter);
Measurement extract_measurement(const std::string& filename);
std::vector<Measurement> extract_measurements(const std::string& filename, int n_meas);
std::vector<Measurement> load_and_initialize_data(const std::string& path, int num_measurements);
cv::Mat load_world_points(const std::string& filename);
bool contains(const std::vector<int>& vec, int value);
float match_error(const Measurement& m1, 
                 const Measurement& m2, 
                 const std::vector<Eigen::Vector2i>& matched_ids);

// Conversion Functions
pr::Vector3fVector matToV3fV(const cv::Mat& mat);
pr::Vector2fVector data_point_v_to_v2fv(const std::vector<Data_Point>& data_points);
pr::IntPairVector imgToWorldCorrespondences(const cv::Mat& world_points, const std::vector<Eigen::Vector2i>& matched_ids_meas);
Eigen::Isometry3f createIsometryFromPose(const Eigen::Vector3f& poseVector);
Eigen::Isometry3f createIsometryFromRt(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& translationVector);
Eigen::Vector3f cvToEigenVector(const cv::Mat& inputMat);
Eigen::Matrix3f cvToEigenMatrix(const cv::Mat& inputMat);

// Debugging Functions
void printIsometry(const Eigen::Isometry3f& iso);
void computeRelativeMotionError(const Eigen::Isometry3f& rel_T, const Eigen::Isometry3f& rel_GT);
void visualizeMatches(const cv::Mat& matched_points1, const cv::Mat& matched_points2);
Eigen::Isometry3f oneRound(Eigen::Isometry3f last_pose_estimate, 
                           pr::Camera& pr_cam, 
                           const pr::Vector3fVector& world_points, 
                           const pr::Vector2fVector& image_points,
                           const pr::IntPairVector& correspondences);
void printVector3fVector(const Vector3fVector& vec);
void printVector2i(const std::vector<Eigen::Vector2i>& vec);

// Filtering Functions
void filter_correspondences(
    const std::vector<std::vector<Eigen::Vector2i>>& all_matched_ids_meas, 
    std::vector<Eigen::Vector2i>& common_correspondences, 
    std::vector<Eigen::Vector2i>& new_correspondences,
    const cv::Mat& previous_world_points,
    cv::Mat& new_world_points
);
void filter_matches(const std::vector<Data_Point>& img_points1,
                    const std::vector<Data_Point>& img_points2,
                    const std::vector<Eigen::Vector2i>& correspondences,
                    std::vector<Data_Point>& filtered_img_points1,
                    std::vector<Data_Point>& filtered_img_points2);

// Conversion Utilities
pr::IntPairVector eigenToIntPairVector(const std::vector<Eigen::Vector2i>& eigenVector);
Eigen::Isometry3f augment_pose(const Eigen::Vector3f& pose);

// Overloaded Operators for Debugging
std::ostream& operator<<(std::ostream& os, const Eigen::Isometry3f& iso);
std::ostream& operator<<(std::ostream& os, const std::vector<Eigen::Isometry3f>& vec);
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec);

// Matching Utilities
int count_correct_matches(const std::vector<Eigen::Vector2i>& matched_ids_real);

// Pose Utilities
cv::Mat isometry3fToCvMat(const Eigen::Isometry3f& isometry);

// Scaling Functions
float compute_scale(const std::vector<Eigen::Vector3f>& points_reconstructed, 
                   const std::vector<Eigen::Vector3f>& points_ground_truth);
float compute_scale(const cv::Mat& points1, const cv::Mat& points2);

// World Points Extraction and Rescaling
cv::Mat extract_world_points(const cv::Mat& world_points, 
                             const std::vector<Eigen::Vector2i>& correspondences);
cv::Mat rescale_points(const cv::Mat& points, float scale);

// Visualization Function
void create_plot(const std::vector<Eigen::Isometry3f>& gt_poses, 
                const std::vector<Eigen::Isometry3f>& est_poses, 
                const std::string& title);

template <typename T, typename U>
std::vector<U> vec_map(const std::vector<T>& vec, U T::*member);
float computeRotationError(const Eigen::Matrix3f &R_err);