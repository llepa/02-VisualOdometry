#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <opencv2/viz.hpp>
#include <Eigen/Dense>
#include "defs.h"
#include "camera.h"

using namespace std;
using namespace Eigen;

struct Measurement {
    int seq;
    Eigen::Vector3f gt_pose;
    Eigen::Vector3f odometry_pose;
    std::vector<int> ids_meas;
    std::vector<int> ids_real;
    cv::Mat points2D;
    std::vector<Eigen::VectorXf> appearances;
};

float euclidean_distance(const VectorXf& vec1, const VectorXf& vec2);

string size(cv::Mat m);

void match_points(const std::vector<Measurement>& measurements, 
                  const int idx1, 
                  const int idx2, 
                  cv::Mat& matches1, 
                  cv::Mat& matches2, 
                  std::vector<Eigen::Vector2i>& matched_ids_meas,
                  std::vector<Eigen::Vector2i>& matched_ids_real);

vector<string> split(const string& str, const string& delimiter);

Measurement extract_measurement(const string& filename);

vector<Measurement> extract_measurements(const string& filename, int n_meas);

vector<Measurement> load_and_initialize_data(const string& path, int num_measurements);

bool contains(const std::vector<int>& vec, int value);

float match_error(const Measurement& m1, 
                  const Measurement& m2, 
                  const std::vector<Eigen::Vector2i>& matched_ids);

pr::Vector3fVector matToV3fV_Type21(const cv::Mat& mat);

pr::Vector3fVector matToV3fV(const cv::Mat& mat);

pr::Vector2fVector matToV2fV(const cv::Mat& mat);

pr::IntPairVector imgToWorldCorrespondences(cv::Mat world_points, std::vector<Eigen::Vector2i>& matched_ids_meas);

Eigen::Isometry3f createIsometryFromPose(Eigen::Vector3f poseVector);

Eigen::Isometry3f createIsometryFromRt(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& translationVector);

Eigen::Vector3f cvToEigenVector(const cv::Mat& inputMat);

Eigen::Matrix3f cvToEigenMatrix(const cv::Mat& inputMat);

void printIsometry(const Eigen::Isometry3f& iso);

void computeRelativeMotionError(const Eigen::Isometry3f& rel_T, const Eigen::Isometry3f& rel_GT);

void visualizeMatches(cv::Mat matched_points1, cv::Mat matched_points2);

pr::Camera oneRound(pr::Camera pr_cam, pr::Vector3fVector world_points, pr::Vector2fVector image_points, pr::IntPairVector correspondences);

void printVector3fVector(const pr::Vector3fVector& vec);

void printVector2i(const std::vector<Eigen::Vector2i>& vec);

void filter_correspondences(
    const std::vector<std::vector<Eigen::Vector2i>>& all_matched_ids_meas, 
    std::vector<Eigen::Vector2i>& common_correspondences, 
    std::vector<Eigen::Vector2i>& new_correspondences,
    const cv::Mat& previous_world_points,
    cv::Mat& new_world_points
);