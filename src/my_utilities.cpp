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
#include "picp_solver.h"
#include "defs.h"

using namespace std;
using namespace Eigen;

float euclidean_distance(const VectorXf& vec1, const VectorXf& vec2) {
    return (vec1 - vec2).norm();
}

string size(cv::Mat m) {
    return "[" + std::to_string(m.rows) + " x " + std::to_string(m.cols) + "]";
}

void match_points(const std::vector<Measurement>& measurements, 
                  const int idx1, 
                  const int idx2, 
                  cv::Mat& matches1, 
                  cv::Mat& matches2, 
                  std::vector<Eigen::Vector2i>& matched_ids_meas,
                  std::vector<Eigen::Vector2i>& matched_ids_real) {
    if (idx1 < 0 || idx1 >= measurements.size() || idx2 < 0 || idx2 >= measurements.size()) {
        std::cerr << "Invalid indices for measurements." << std::endl;
        return;
    }

    Measurement m1 = measurements[idx1];
    Measurement m2 = measurements[idx2];
    float threshold = 0.7;

    // std::cout << "Finding best matches for each point..." << std::endl;

    for (int i = 0; i < m1.appearances.size(); i++) {
        float min_distance = std::numeric_limits<float>::max();
        float second_min_distance = std::numeric_limits<float>::max();
        int min_idx = -1;

        for (int j = 0; j < m2.appearances.size(); j++) {
            float distance = (m1.appearances[i] - m2.appearances[j]).norm();
            
            if (distance < min_distance) {
                second_min_distance = min_distance;
                min_distance = distance;
                min_idx = j;
            } else if (distance < second_min_distance) {
                second_min_distance = distance;
            }
        }

        float ratio = min_distance / second_min_distance;

        if (ratio < 0.8) {  // Assuming 0.8 as the threshold; can be adjusted.
            // Consider this a good match and process it as before
            if (min_distance <= threshold) {
                matches1.push_back(m1.points2D.row(i));
                matches2.push_back(m2.points2D.row(min_idx));

                Eigen::Vector2i match_ids_meas;
                match_ids_meas(0) = m1.ids_meas[i];
                match_ids_meas(1) = m2.ids_meas[min_idx];
                matched_ids_meas.push_back(match_ids_meas);

                Eigen::Vector2i match_ids_real;
                match_ids_real(0) = m1.ids_real[i];
                match_ids_real(1) = m2.ids_real[min_idx];
                matched_ids_real.push_back(match_ids_real);
            }
        }
    }
        // std::cout << "Finished finding matches" << std::endl;

}

vector<string> split(const string& str, const string& delimiter) {
    vector<string> tokens;
    string token;
    size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != string::npos) {
        token = str.substr(start, end - start);
        if (!token.empty()) tokens.push_back(token);
        start = end + delimiter.length();
    }
    token = str.substr(start);
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

Measurement extract_measurement(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        exit(EXIT_FAILURE);
    }
    Measurement meas;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> tokens = split(line, " ");
        
        if (tokens[0] == "seq:") {
            meas.seq = stoi(tokens[1]);
        } else if (tokens[0] == "gt_pose:") {
            meas.gt_pose[0] = stof(tokens[1]);
            meas.gt_pose[1] = stof(tokens[2]);
            meas.gt_pose[2] = stof(tokens[3]);
        } else if (tokens[0] == "odom_pose:") {
            meas.odometry_pose[0] = stof(tokens[1]);
            meas.odometry_pose[1] = stof(tokens[2]);
            meas.odometry_pose[2] = stof(tokens[3]);
        } else if (tokens[0] == "point") {
            if (tokens.size() < 15) {
                cerr << "Error: not enough tokens for measurement " << endl;
                continue;
            }
            // add token 1 to meas.id_meas and token 2 to id_real
            meas.ids_meas.push_back(stoi(tokens[1]));
            meas.ids_real.push_back(stoi(tokens[2]));

            // add tokens from 3 to 4 to points2D
            cv::Mat point(1, 2, CV_32F);
            point.at<float>(0, 0) = stof(tokens[3]);
            point.at<float>(0, 1) = stof(tokens[4]);
            meas.points2D.push_back(point);

            // add tokens from 5 to 14 to appearances
            Eigen::VectorXf appearance(10);
            for (int i = 5; i < 15; i++) {
                appearance[i-5] = stof(tokens[i]);
            }
            meas.appearances.push_back(appearance);

        } else {
            std::cerr << "Invalid line in file " << filename << ": " << line << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    file.close();
    return meas;
}

vector<Measurement> extract_measurements(const string& filename, int n_meas) {
    cout << "Extracting measurements from files..." << endl;
    vector<Measurement> measurements;

    for (int i=0; i<=n_meas; i++) {  
        // open measurement file
        stringstream ss;
        ss << filename << setfill('0') << setw(5) << i << ".dat";
        string filename = ss.str();
        Measurement meas = extract_measurement(filename);
        measurements.push_back(meas);
    }
    cout << "Measurements extracted" << endl;
    return measurements;
}

vector<Measurement> load_and_initialize_data(const string& path, int num_measurements) {
    return extract_measurements(path, num_measurements);
}

bool contains(const std::vector<int>& vec, int value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

float match_error(const Measurement& m1, const Measurement& m2, const std::vector<Eigen::Vector2i>& matched_ids) {
    // mismatches
    int mismatches = 0;
    int potential_matches = 0; 
    for (size_t i = 0; i < matched_ids.size(); ++i) {
        if (matched_ids[i](0) != matched_ids[i](1)) {
            mismatches++;
        }
    }
    // missed matches
    for (size_t i = 0; i < m1.ids_real.size(); ++i) {
        if (contains(m2.ids_real, m1.ids_real[i])) {
            potential_matches += 1;
        }
    }

    int missed = potential_matches - (matched_ids.size() - mismatches);

    float total_error = static_cast<float>(mismatches + missed) / potential_matches;
    std::cout << "Missed matches: " << missed << std::endl;
    std::cout << "Mismatches: " << mismatches << std::endl;

    return total_error;
}


pr::Vector3fVector matToV3fV_Type21(const cv::Mat& mat) {
    // std::cout << "Type: " << mat.type() << std::endl;
    // std::cout << "Size: " << mat.size() << std::endl;
    // std::cout << "Cols: " << mat.cols << std::endl;
    
    // Ensure the matrix has 3D points (multiple of 3 columns)
    if (mat.type() != CV_32FC3) {
        throw std::runtime_error("Input matrix does not contain a multiple of 3 elements or is of an incorrect type.");
    }

    pr::Vector3fVector result;

    // Iterate through the matrix by steps of 3 to create each Vector3f
    for (int i = 0; i < mat.rows; i += 1) {
        Eigen::Vector3f point(
            mat.at<cv::Vec3f>(0, i)[0],
            mat.at<cv::Vec3f>(0, i)[1],
            mat.at<cv::Vec3f>(0, i)[2]
            );
        result.push_back(point);  // Add the 3D point to the result vector
    }

    return result;
}


pr::Vector3fVector matToV3fV(const cv::Mat& mat) {
    // std::cout << "Type: " << mat.type() << std::endl;
    // std::cout << "Size: " << mat.size() << std::endl;

    if (mat.type() != CV_32F || mat.cols != 3) {
        throw std::runtime_error("Input matrix is not of appropriate type or dimensions.");
    }

    pr::Vector3fVector result;
    for (int i = 0; i < mat.rows; ++i) {
        result.push_back(Eigen::Vector3f(mat.at<float>(i, 0), mat.at<float>(i, 1), mat.at<float>(i, 2)));
    }

    return result;
}

pr::Vector2fVector matToV2fV(const cv::Mat& mat) {
    if (mat.type() != CV_32F || mat.cols != 2) {
        throw std::runtime_error("Input matrix is not of appropriate type or dimensions.");
    }

    pr::Vector2fVector result;
    for (int i = 0; i < mat.rows; ++i) {
        result.push_back(Eigen::Vector2f(mat.at<float>(i, 0), mat.at<float>(i, 1)));
    }

    return result;
}

pr::IntPairVector imgToWorldCorrespondences(cv::Mat world_points, std::vector<Eigen::Vector2i>& matched_ids_meas) {
    pr::IntPairVector correspondences;
    for (int i = 0; i < matched_ids_meas.size(); i++) {
        correspondences.push_back(pr::IntPair(matched_ids_meas[i](1), i));
    }
    return correspondences;
}

Eigen::Isometry3f createIsometryFromPose(Eigen::Vector3f poseVector) {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translation() = Eigen::Vector3f(poseVector[0], poseVector[1], 0);
    pose.rotate(Eigen::AngleAxisf(poseVector[2], Eigen::Vector3f::UnitZ()));
    return pose;
}

Eigen::Isometry3f createIsometryFromRt(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& translationVector) {
    Eigen::Isometry3f iso = Eigen::Isometry3f::Identity();
    iso.linear() = rotationMatrix;
    iso.translation() = translationVector;
    return iso;
}

Eigen::Vector3f cvToEigenVector(const cv::Mat& inputMat) {
    if (inputMat.rows != 3 || inputMat.cols != 1) {
        throw std::invalid_argument("Input cv::Mat must be a 1x3 vector of type CV_32F.");
    }
    cv::Mat mat;
    inputMat.convertTo(mat, CV_32F);  

    Eigen::Vector3f eigenVec;
    eigenVec << mat.at<float>(0, 0), 
                mat.at<float>(0, 1), 
                mat.at<float>(0, 2);

    return eigenVec;
}

Eigen::Matrix3f cvToEigenMatrix(const cv::Mat& inputMat) {
    if (inputMat.rows != 3 || inputMat.cols != 3) {
        throw std::invalid_argument("Input cv::Mat must be a 3x3 matrix.");
    }
    cv::Mat mat;
    inputMat.convertTo(mat, CV_32F);
    Eigen::Matrix3f eigenMat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigenMat(i, j) = mat.at<float>(i, j);
        }
    }
    return eigenMat;
}

void printIsometry(const Eigen::Isometry3f& iso) {
    Eigen::Matrix3f rotation = iso.rotation();
    Eigen::Vector3f translation = iso.translation();

    std::cout << "------- World In Camera Pose -------" << std::endl;
    std::cout << "Rotation Matrix: \n" << rotation << std::endl;
    std::cout << "Translation Vector: \n" << translation.transpose() << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

void computeRelativeMotionError(const Eigen::Isometry3f& rel_T, const Eigen::Isometry3f& rel_GT) {
    // Compute the relative error transformation: error_T = inv(rel_T) * rel_GT
    Eigen::Isometry3f error_T = rel_T.inverse() * rel_GT;

    // Rotation error: trace(I - error_T(1:3, 1:3))
    Eigen::Matrix3f error_R = error_T.rotation(); // Extract the 3x3 rotation matrix
    float rotation_error = (Eigen::Matrix3f::Identity() - error_R).trace();

    // Translation error: norm of translation parts, compute scale ratio
    Eigen::Vector3f trans_rel_T = rel_T.translation(); // Extract translation from rel_T
    Eigen::Vector3f trans_rel_GT = rel_GT.translation(); // Extract translation from rel_GT
    float norm_rel_T = trans_rel_T.norm();
    float norm_rel_GT = trans_rel_GT.norm();
    float scale_ratio = norm_rel_T / norm_rel_GT;

    // Output the results
    std::cout << "Rotation Error: " << rotation_error << std::endl;
    std::cout << "Scale Ratio: " << scale_ratio << std::endl;
}

void visualizeMatches(cv::Mat matched_points1, cv::Mat matched_points2) {
    // Assuming matched_points1 and matched_points2 are your keypoints.
    std::vector<cv::KeyPoint> keypoints1, keypoints2;

    
    // Convert 2D points to KeyPoint format
    for (int i = 0; i < matched_points1.rows; ++i) {
        keypoints1.push_back(cv::KeyPoint(matched_points1.at<cv::Point2f>(i), 1));
        keypoints2.push_back(cv::KeyPoint(matched_points2.at<cv::Point2f>(i), 1));
    }

    // Create synthetic images
    cv::Mat img1 = cv::Mat::zeros(480, 640, CV_8UC3); 
    cv::Mat img2 = cv::Mat::zeros(480, 640, CV_8UC3); 

    // Draw keypoints
    for (const auto& kp : keypoints1) {
        cv::circle(img1, kp.pt, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto& kp : keypoints2) {
        cv::circle(img2, kp.pt, 2, cv::Scalar(0, 0, 255), -1);
    }

    std::vector<cv::DMatch> matches; 
    for (int i = 0; i < keypoints1.size(); ++i) {
        matches.push_back(cv::DMatch(i, i, 0));
    }

    // Visualize matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    cv::imshow("Matches", img_matches);
    cv::waitKey();
}

pr::Camera oneRound(pr::Camera pr_cam, pr::Vector3fVector world_points, pr::Vector2fVector image_points, pr::IntPairVector correspondences) {
    int maxIterations = 100;
    double convergenceThreshold = 0.00001;
    double prevError = std::numeric_limits<double>::max();
    double currentError;

    pr::PICPSolver solver = pr::PICPSolver();
    solver.init(pr_cam, world_points, image_points);

    std::cout << "\n" << std::endl;
    for (int i = 0; i < maxIterations; i++) {
        solver.oneRound(correspondences, false);
        currentError = solver.chiInliers();
        
        // std::cout << "----------------------------------------------------" << "\n";
        // std::cout << "Round " << i+1 << "\n";
        // std::cout << "inliers: " << solver.numInliers() << "/" << correspondences.size() << endl;
        // std::cout << "error (inliers): " << currentError << " (outliers): " << solver.chiOutliers() << endl;
        // std::cout << "----------------------------------------------------" << "\n";

        if (abs(prevError - currentError) < convergenceThreshold) {
            break;
        }
        prevError = currentError;
    }
    std::cout << "error (inliers): " << currentError << " (outliers): " << solver.chiOutliers() << endl;
    return solver.camera();

    // std::cout << "\n" << std::endl;
    // printIsometry(pr_cam.worldInCameraPose());
}

void printVector3fVector(const pr::Vector3fVector& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        const Eigen::Vector3f& v = vec[i];
        std::cout << "Vector " << i << ": (" 
                  << v[0] << ", " 
                  << v[1] << ", " 
                  << v[2] << ")" 
                  << std::endl;
    }
}

void printVector2i(const std::vector<Eigen::Vector2i>& vec) {
    std::cout << "Vector contents:" << std::endl;
    for (const auto& v : vec) {
        std::cout << "[" << v.x() << ", " << v.y() << "]" << std::endl;
    }
}


 void filter_correspondences(
    const std::vector<std::vector<Eigen::Vector2i>>& all_matched_ids_meas, 
    std::vector<Eigen::Vector2i>& common_correspondences, 
    std::vector<Eigen::Vector2i>& new_correspondences,
    const cv::Mat& previous_world_points,
    cv::Mat& new_world_points
) {
    // Get the latest and previous correspondences from the list
    const std::vector<Eigen::Vector2i>& latest_correspondences = all_matched_ids_meas.back();
    const std::vector<Eigen::Vector2i>& previous_correspondences = all_matched_ids_meas[all_matched_ids_meas.size() - 2];

    for (const Eigen::Vector2i& latest_correspondence : latest_correspondences) {
        auto el = std::find_if(previous_correspondences.begin(), previous_correspondences.end(),
                               [latest_correspondence](const Eigen::Vector2i& vec) {
                                   return vec.y() == latest_correspondence.x();
                               });

        if (el != previous_correspondences.end()) {
            common_correspondences.push_back(*el);
            
            // Calculate the index of `el` within `previous_correspondences`
            int idx = std::distance(previous_correspondences.begin(), el);
            std::cout << "idx: " << idx << std::endl;

            // Use `idx` to access the corresponding point in `previous_world_points`
            new_world_points.push_back(previous_world_points.row(idx));
        } else {
            new_correspondences.push_back(latest_correspondence);
        }
    }
    // std::cout << "world_points: " << new_world_points << std::endl;
    std::cout << "new_world_points size: " << new_world_points.size() << std::endl;
}


