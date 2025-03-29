#include "my_utilities.h"

#include "defs.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <vector>
#include <algorithm>
#include <limits>


using namespace pr;

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    // Implementation remains the same as provided
    std::vector<std::string> tokens;
    std::string token;
    size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        token = str.substr(start, end - start);
        if (!token.empty()) tokens.push_back(token);
        start = end + delimiter.length();
    }
    token = str.substr(start);
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

Measurement extract_measurement(const std::string& filename) {
    // Implementation remains the same as provided
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    Measurement meas;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> tokens = split(line, " ");

        if (tokens.empty()) continue;

        if (tokens[0] == "seq:") {
            if (tokens.size() < 2) {
                std::cerr << "Error: 'seq:' line has insufficient tokens in file " << filename << std::endl;
                continue;
            }
            meas.seq = std::stoi(tokens[1]);
        } 
        else if (tokens[0] == "gt_pose:") {
            if (tokens.size() < 4) {
                std::cerr << "Error: 'gt_pose:' line has insufficient tokens in file " << filename << std::endl;
                continue;
            }
            meas.gt_pose[0] = std::stof(tokens[1]);
            meas.gt_pose[1] = std::stof(tokens[2]);
            meas.gt_pose[2] = std::stof(tokens[3]);
        } 
        else if (tokens[0] == "odom_pose:") {
            if (tokens.size() < 4) {
                std::cerr << "Error: 'odom_pose:' line has insufficient tokens in file " << filename << std::endl;
                continue;
            }
            meas.odometry_pose[0] = std::stof(tokens[1]);
            meas.odometry_pose[1] = std::stof(tokens[2]);
            meas.odometry_pose[2] = std::stof(tokens[3]);
        } 
        else if (tokens[0] == "point") {
            if (tokens.size() < 15) {
                std::cerr << "Error: 'point' line has insufficient tokens in file " << filename << std::endl;
                continue;
            }

            // Parse measurement and real IDs
            int id_meas = std::stoi(tokens[1]);
            int id_real = std::stoi(tokens[2]);

            // Parse 2D image coordinates
            float x = std::stof(tokens[3]);
            float y = std::stof(tokens[4]);
            cv::Point2f coordinates(x, y);

            // Parse descriptor (tokens[5] to tokens[14])
            Eigen::VectorXf descriptor(10); // Assuming descriptor size is 10
            for (int i = 5; i < 15; ++i) {
                descriptor[i - 5] = std::stof(tokens[i]);
            }

            // Create a Data_Point instance and add it to data_points
            Data_Point dp(id_meas, id_real, coordinates, descriptor);
            meas.data_points.push_back(dp);
        } 
        else {
            std::cerr << "Invalid line in file " << filename << ": " << line << std::endl;
            // Continue processing other lines instead of exiting
            continue;
        }
    }

    file.close();
    return meas;
}


std::vector<Measurement> extract_measurements(const std::string& filename, int n_meas) {
    std::cout << "Extracting measurements from files..." << std::endl;
    std::vector<Measurement> measurements;

    for (int i = 0; i < n_meas; ++i) {
        // Construct the filename with leading zeros (e.g., meas-00000.dat)
        std::stringstream ss;
        ss << filename << std::setfill('0') << std::setw(5) << i << ".dat";
        std::string current_filename = ss.str();
        Measurement meas = extract_measurement(current_filename);
        measurements.push_back(meas);
    }
    std::cout << "Measurements extracted" << std::endl;
    return measurements;
}


std::vector<Measurement> load_and_initialize_data(const std::string& path_prefix, int num_measurements) {
    return extract_measurements(path_prefix, num_measurements);
}


std::vector<World_Point> load_world_points(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening world points file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<World_Point> world_points;
    std::string line;
    const int descriptor_length = 10;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int id;
        if (!(iss >> id)) {
            std::cerr << "Error reading id from line: " << line << std::endl;
            continue;
        }

        float x, y, z;
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Error reading coordinates from line: " << line << std::endl;
            continue;
        }
        cv::Point3f coord(x, y, z);

        Eigen::VectorXf descriptor(descriptor_length);
        bool descriptor_complete = true;
        for (int i = 0; i < descriptor_length; ++i) {
            if (!(iss >> descriptor[i])) {
                std::cerr << "Warning: Incomplete descriptor in line: " << line << std::endl;
                descriptor_complete = false;
                break;
            }
        }
        if (!descriptor_complete) {
            continue; // Skip lines with incomplete descriptor data
        }

        World_Point point(coord, descriptor, id);
        world_points.push_back(point);
    }

    file.close();
    return world_points;
}


void extract_coordinates_from_matches(
    std::vector<std::pair<Data_Point, Data_Point>> matches,
    std::vector<cv::Point2f>& matches1,
    std::vector<cv::Point2f>& matches2
){
    matches1.reserve(matches.size());
    matches2.reserve(matches.size());
    for (const auto& match : matches) {
        matches1.push_back(match.first.coordinates);
        matches2.push_back(match.second.coordinates);
    }
}


cv::Mat extract_matrix_coordinates(const std::vector<World_Point>& points) {
    cv::Mat pts1(static_cast<int>(points.size()), 3, CV_32F);
    for (size_t i = 0; i < points.size(); i++) {
        pts1.at<float>(static_cast<int>(i), 0) = points[i].coordinates.x;
        pts1.at<float>(static_cast<int>(i), 1) = points[i].coordinates.y;
        pts1.at<float>(static_cast<int>(i), 2) = points[i].coordinates.z;
    }
    return pts1;
}

Vector2fVector extract_V2fV(const std::vector<Data_Point>& points) {
    Vector2fVector result;
    for (const auto& point : points) {
        result.emplace_back(Eigen::Vector2f(point.coordinates.x, point.coordinates.y));
    }
    return result;
}

Vector3fVector extract_V3fV(const std::vector<World_Point>& points) {
    Vector3fVector result;
    for (const auto& point : points) {
        result.emplace_back(Eigen::Vector3f(point.coordinates.x, point.coordinates.y, point.coordinates.z));
    }
    return result;
}


float compute_scale(const std::vector<Eigen::Vector3f>& points_reconstructed, 
                   const std::vector<Eigen::Vector3f>& points_ground_truth) {
    float total_scale = 0.0f;
    int valid_points = 0;
    
    for (size_t i = 0; i < points_reconstructed.size() && i < points_ground_truth.size(); ++i) {
        float reconstructed_dist = points_reconstructed[i].norm();
        float ground_truth_dist = points_ground_truth[i].norm();
        
        if (reconstructed_dist > 0 && ground_truth_dist > 0) {
            total_scale += ground_truth_dist / reconstructed_dist;
            valid_points++;
        }
    }
    
    return (valid_points > 0) ? (total_scale / valid_points) : 1.0f;
}


Eigen::Isometry3f augment_pose(const Eigen::Vector3f& pose) {
    Eigen::Isometry3f gt_T = Eigen::Isometry3f::Identity();
    float theta = pose.coeff(2);
    Eigen::Matrix3f rotationMatrix;
    rotationMatrix << std::cos(theta), -std::sin(theta), 0,
                      std::sin(theta),  std::cos(theta), 0,
                      0,                0,               1;

    gt_T.linear() = rotationMatrix;

    // Assign the first two components of the translation and set the third to 0.
    gt_T.translation().head<2>() = pose.head<2>();
    gt_T.translation()(2) = 0.0f;

    return gt_T;
}


Eigen::Isometry3f oneRound(Eigen::Isometry3f last_pose_estimate,
                           pr::Camera& pr_cam,
                           const pr::Vector3fVector& world_points,
                           const pr::Vector2fVector& image_points,
                           const pr::IntPairVector& correspondences) {
    // Early return if not enough correspondences
    if (correspondences.size() < 10) {
        std::cerr << "Warning: Not enough correspondences for pose estimation ("
                  << correspondences.size() << " < 10), using previous pose" << std::endl;
        return last_pose_estimate;
    }

    // Initialize the solver and camera pose
    pr::PICPSolver solver;
    pr_cam.setWorldInCameraPose(last_pose_estimate);
    solver.init(pr_cam, world_points, image_points);
    solver.setKernelThreshold(100.0f);
    std::cout << "Kernel threshold set to 100.0f" << std::endl;

    // Optimization parameters
    const int maxIterations = 50;
    double prevError = std::numeric_limits<double>::max();
    double currentError = prevError;
    const double convergenceThreshold = 0.05;

    // Single-stage iterative optimization with logging
    for (int i = 0; i < maxIterations; ++i) {
        if (!solver.oneRound(correspondences, true)) {
            std::cerr << "Solver iteration " << i << " failed." << std::endl;
            break;
        }

        currentError = solver.chiInliers();
        double relImprovement = (prevError > 1e-10) ? std::abs(prevError - currentError) / prevError : 0.0;

        std::cout << "Iteration " << i
                  << ", current error: " << currentError
                  << ", relative improvement: " << relImprovement
                  << ", inliers: " << solver.numInliers() << std::endl;

        if (relImprovement < convergenceThreshold) {
            std::cout << "Convergence reached at iteration " << i << std::endl;
            break;
        }
        prevError = currentError;
    }

    Eigen::Isometry3f finalPose = solver.camera().worldInCameraPose();
    std::cout << "Final pose computed. Inliers: " << solver.numInliers()
              << " out of " << correspondences.size() << std::endl;

    return finalPose;
}


void create_plot(const std::vector<Eigen::Isometry3f>& gt_poses,
                const std::vector<Eigen::Isometry3f>& est_poses,
                const std::string& title) {
    // Create a white canvas
    cv::Mat trajectory(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    // Determine scaling and translation for visualization
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // Combine both ground truth and estimated poses to determine scaling bounds
    for (const auto& pose : gt_poses) {
        min_x = std::min(min_x, pose.translation().x());
        max_x = std::max(max_x, pose.translation().x());
        min_y = std::min(min_y, pose.translation().y());
        max_y = std::max(max_y, pose.translation().y());
    }
    for (const auto& pose : est_poses) {
        min_x = std::min(min_x, pose.translation().x());
        max_x = std::max(max_x, pose.translation().x());
        min_y = std::min(min_y, pose.translation().y());
        max_y = std::max(max_y, pose.translation().y());
    }

    // Handle cases where all poses have the same x or y
    if (max_x - min_x == 0) max_x += 1.0f;
    if (max_y - min_y == 0) max_y += 1.0f;

    // Scaling factor to fit trajectories within the canvas
    float scale_x = 500.0f / (max_x - min_x);
    float scale_y = 500.0f / (max_y - min_y);
    float scale = std::min(scale_x, scale_y);

    // Function to map poses to 2D points
    auto map_poses_to_2d = [&](const std::vector<Eigen::Isometry3f>& poses) {
        std::vector<cv::Point2f> points2d;
        points2d.reserve(poses.size());
        for (const auto& pose : poses) {
            int x = static_cast<int>(50 + (pose.translation().x() - min_x) * scale);
            int y = static_cast<int>(550 - (pose.translation().y() - min_y) * scale); // Flip y-axis for image coordinates
            points2d.emplace_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        }
        return points2d;
    };

    // Map GT and estimated poses to 2D points
    std::vector<cv::Point2f> gt_points2d = map_poses_to_2d(gt_poses);
    std::vector<cv::Point2f> est_points2d = map_poses_to_2d(est_poses);

    // Draw GT trajectory (blue)
    for (size_t i = 1; i < gt_points2d.size(); ++i) {
        cv::line(trajectory, gt_points2d[i-1], gt_points2d[i], cv::Scalar(255, 0, 0), 2);
    }

    // Draw estimated trajectory (red)
    for (size_t i = 1; i < est_points2d.size(); ++i) {
        cv::line(trajectory, est_points2d[i-1], est_points2d[i], cv::Scalar(0, 0, 255), 2);
    }

    // Add markers for first and last points of GT (green and blue)
    if (!gt_points2d.empty()) {
        cv::circle(trajectory, gt_points2d.front(), 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(trajectory, gt_points2d.back(), 5, cv::Scalar(255, 0, 0), -1);
    }

    // Add markers for first and last points of estimated trajectory (green and red)
    if (!est_points2d.empty()) {
        cv::circle(trajectory, est_points2d.front(), 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(trajectory, est_points2d.back(), 5, cv::Scalar(0, 0, 255), -1);
    }

    // Add title
    cv::putText(trajectory, title, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Show the plot
    cv::imshow(title, trajectory);
    cv::waitKey(0);
}

float computeRotationError(const Eigen::Matrix3f &R_err) {
    // Calculate the cosine of the rotation angle using the trace of R_err.
    float cos_val = (R_err.trace() - 1.0f) / 2.0f;

    // Clamp cos_val to the range [-1, 1] to avoid NaNs from acos.
    cos_val = std::max(-1.0f, std::min(1.0f, cos_val));

    // Compute the angle in radians, then convert to degrees.
    float radiants = std::acos(cos_val);
    return radiants;
}


std::vector<std::pair<Data_Point, Data_Point>> add_new_world_points(
    std::vector<std::pair<Data_Point, World_Point>> img_world_matches,
    std::vector<std::pair<Data_Point, Data_Point>> img_matches
) {
    std::vector<std::pair<Data_Point, Data_Point>> points_to_triangulate;

    for (const auto& img_match : img_matches) {
        bool found = false;
        for (const auto& img_world_match : img_world_matches) {
            if (img_match.second.id_meas == img_world_match.first.id_meas) {
                found = true;
                break;
            }
        }
        if (!found) {
            points_to_triangulate.push_back(img_match);
        }
    }

    std::cout << "Points to be triangulated: " << points_to_triangulate.size() << std::endl;
    return points_to_triangulate;
}


int check_world_points_sanity(const std::vector<World_Point>& world_points) {
    std::vector<int> presences(1000, 0);
    std::vector<std::pair<int, int>> duplicates;

    for (const auto& point : world_points) {
        // Assuming point.id_real is within the range [0, 999]
        presences[point.id_real]++;
    }

    int duplicate_count = 0;
    for (size_t i = 0; i < presences.size(); i++) {
        int p = presences[i];
        if (p > 1) {
            duplicate_count++;
            duplicates.push_back(std::make_pair(static_cast<int>(i), p));
        }
    }
    std::cout << "Number of duplicate world points: " << duplicate_count << std::endl;
    return duplicate_count;
}


Eigen::Affine3f alignTrajectories(const std::vector<Eigen::Isometry3f>& poses,
                                  const std::vector<Eigen::Isometry3f>& gt_poses) {
    assert(poses.size() == gt_poses.size() && "Pose vectors must be of equal size");
    const size_t N = poses.size();

    // Build 3xN matrices containing the translation components.
    Eigen::MatrixXf P(3, N), Q(3, N);
    for (size_t i = 0; i < N; i++) {
        P.col(i) = poses[i].translation();
        Q.col(i) = gt_poses[i].translation();
    }

    // Compute the similarity transform as a 4x4 matrix.
    Eigen::Matrix4f T_mat = Eigen::umeyama(P, Q, true);

    // Explicitly assign the matrix to an Eigen::Affine3f.
    Eigen::Affine3f T;
    T.matrix() = T_mat;
    return T;
}