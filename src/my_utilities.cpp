// my_utilities.cpp

// Avoid using 'using namespace' globally in source files
// Instead, use fully qualified names or specific using declarations within functions

// Struct Definitions (if any additional functions are needed)

// Function Definitions

#include <my_utilities.h>
#include <string>
#include <vector>
/**
 * @brief Splits a string by a given delimiter.
 * 
 * @param str The string to split.
 * @param delimiter The delimiter to split by.
 * @return std::vector<std::string> The resulting tokens.
 */
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

/**
 * @brief Extracts a single measurement from a file.
 * 
 * @param filename The path to the measurement file.
 * @return Measurement The extracted measurement.
 */
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

/**
 * @brief Extracts multiple measurements from a series of files.
 * 
 * @param filename The base path prefix for measurement files.
 * @param n_meas The number of measurements to extract.
 * @return std::vector<Measurement> The vector of extracted measurements.
 */
std::vector<Measurement> extract_measurements(const std::string& filename, int n_meas) {
    std::cout << "Extracting measurements from files..." << std::endl;
    std::vector<Measurement> measurements;

    for (int i = 0; i <= n_meas; ++i) {  
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

/**
 * @brief Loads and initializes measurement data.
 * 
 * @param path_prefix The prefix path for measurement files.
 * @param num_measurements The number of measurements to load.
 * @return std::vector<Measurement> The loaded measurements.
 */
std::vector<Measurement> load_and_initialize_data(const std::string& path_prefix, int num_measurements) {
    return extract_measurements(path_prefix, num_measurements);
}

/**
 * @brief Loads world points from a file.
 * 
 * @param filename The path to the world points file.
 * @return cv::Mat The matrix of world points (Nx3).
 */
cv::Mat load_world_points(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening world points file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> points;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> point;
        float value;

        iss >> value;  // Skip index

        // Take only first 3 values after index
        for (int i = 0; i < 3 && iss >> value; ++i) {
            point.push_back(value);
        }

        if (point.size() == 3) {
            points.push_back(point);
        } else {
            std::cerr << "Warning: Incomplete point data in file " << filename << ": " << line << std::endl;
        }
    }

    file.close();

    // Convert to cv::Mat
    cv::Mat result(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            result.at<float>(i, j) = points[i][j];
        }
    }

    return result;
}

std::vector<cv::Point2f> extract_coordinates(const std::vector<Data_Point>& points) {
    std::vector<cv::Point2f> coordinates;
    coordinates.reserve(points.size());
    for (const auto& point : points) {
        coordinates.push_back(point.coordinates);
    }
    return coordinates;
}

cv::Mat extract_matrix_coordinates(const std::vector<Data_Point>& points) {
    cv::Mat pts1(static_cast<int>(points.size()), 2, CV_32F);
    for (size_t i = 0; i < points.size(); i++) {
        pts1.at<float>(static_cast<int>(i), 0) = points[i].coordinates.x;
        pts1.at<float>(static_cast<int>(i), 1) = points[i].coordinates.y;
    }
    return pts1;
}

void match_points(
    const std::vector<Data_Point>& data_points1,
    const std::vector<Data_Point>& data_points2,
    std::vector<std::pair<Data_Point, Data_Point>>& matches,
    std::vector<Eigen::Vector2i>& matched_ids_meas,
    std::vector<Eigen::Vector2i>& matched_ids_real
) {

    // Matching thresholds (adjust these thresholds to suit your descriptor scale)
    const float distanceThreshold = 0.7f; // Maximum allowed Euclidean distance between descriptors.
    const float ratioThreshold = 0.8f;    // Lowe's ratio test threshold.

    // Loop over every data point in the first set.
    for (size_t i = 0; i < data_points1.size(); i++) {
        const Data_Point& point1 = data_points1[i];
        float min_distance = std::numeric_limits<float>::max();
        float second_min_distance = std::numeric_limits<float>::max();
        int best_match_idx = -1;

        // Ensure there are at least two points in the second set for the ratio test.
        if (data_points2.size() < 2)
            continue;

        // Loop over every data point in the second set.
        for (size_t j = 0; j < data_points2.size(); j++) {
            const Data_Point& point2 = data_points2[j];
            float distance = (point1.descriptor - point2.descriptor).norm();

            if (distance < min_distance) {
                second_min_distance = min_distance;
                min_distance = distance;
                best_match_idx = static_cast<int>(j);
            } else if (distance < second_min_distance) {
                second_min_distance = distance;
            }
        }

        // Guard against division by zero or uninitialized second_min_distance.
        if (second_min_distance == 0.0f || second_min_distance == std::numeric_limits<float>::max())
            continue;

        // Apply Lowe's ratio test.
        float ratio = min_distance / second_min_distance;
        if (ratio < ratioThreshold && min_distance <= distanceThreshold) {
            const Data_Point& best_point = data_points2[best_match_idx];

            // Save the matching pair.
            matches.push_back(std::make_pair(point1, best_point));

            // Record the matching IDs.
            Eigen::Vector2i match_ids_meas(point1.id_meas, best_point.id_meas);
            Eigen::Vector2i match_ids_real(point1.id_real, best_point.id_real);
            matched_ids_meas.push_back(match_ids_meas);
            matched_ids_real.push_back(match_ids_real);
        }
    }
}


/**
 * @brief Checks if a vector contains a specific value.
 * 
 * @param vec The vector to search.
 * @param value The value to find.
 * @return true If the value is found.
 * @return false Otherwise.
 */
bool contains(const std::vector<int>& vec, int value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

/**
 * @brief Computes the matching error between two measurements.
 * 
 * @param m1 The first measurement.
 * @param m2 The second measurement.
 * @param matched_ids The matched ID pairs.
 * @return float The total error.
 */
float match_error(const Measurement& m1,
                  const Measurement& m2,
                  const std::vector<Eigen::Vector2i>& matched_ids
                  ) {
    // Count mismatches
    int mismatches = 0;
    for (const Eigen::Vector2i& match : matched_ids) {
        if (match.coeff(0) != match.coeff(1)) {
            mismatches++;
        }
    }

    // Count potential matches based on ground truth real IDs
    int potential_matches = 0; 
    for (const auto& dp1 : m1.data_points) {
        if (contains(vec_map(m2.data_points, &Data_Point::id_real), dp1.id_real)) { // Adjusted to check real IDs correctly
            potential_matches += 1;
        }
    }

    int missed = potential_matches - (matched_ids.size() - mismatches);

    float total_error = (potential_matches > 0) ? (static_cast<float>(mismatches + missed) / potential_matches) : 0.0f;
    std::cout << "Missed matches: " << missed << std::endl;
    std::cout << "Mismatches: " << mismatches << std::endl;

    return total_error;
}

/**
 * @brief Converts a cv::Mat to a Vector3fVector.
 * 
 * @param mat The input matrix (Nx3, CV_32F).
 * @return pr::Vector3fVector The resulting vector of Eigen::Vector3f.
 */
pr::Vector3fVector matToV3fV(const cv::Mat& mat) {
    // Validate matrix type and size
    if (mat.type() != CV_32F || mat.cols != 3) {
        throw std::runtime_error("Input matrix is not of appropriate type or dimensions.");
    }

    pr::Vector3fVector result;
    result.reserve(mat.rows); // Reserve space for efficiency

    for (int i = 0; i < mat.rows; ++i) {
        result.emplace_back(Eigen::Vector3f(mat.at<float>(i, 0), mat.at<float>(i, 1), mat.at<float>(i, 2)));
    }

    return result;
}

/**
 * @brief Converts a cv::Mat to a Vector2fVector.
 * 
 * @param mat The input matrix (Nx2, CV_32F).
 * @return Vector2fVector The resulting vector of Eigen::Vector2f.
 */
pr::Vector2fVector data_point_v_to_v2fv(const std::vector<Data_Point>& data_points) {
    pr::Vector2fVector vec;
    for (const auto& point : data_points) {
        vec.push_back(Eigen::Vector2f(point.coordinates.x, point.coordinates.y));
    }
    return vec;
}

/**
 * @brief Converts matched measurement IDs to world correspondences.
 * 
 * @param matched_ids_meas The matched measurement ID pairs.
 * @return pr::IntPairVector The corresponding world ID pairs.
 */
pr::IntPairVector imgToWorldCorrespondences(const std::vector<Eigen::Vector2i>& matched_ids_meas) {
    pr::IntPairVector correspondences;
    correspondences.reserve(matched_ids_meas.size());

    for (const auto& pair : matched_ids_meas) {
        correspondences.emplace_back(pair(1), pair(0)); // Assuming (world_id, image_id)
    }
    return correspondences;
}

/**
 * @brief Creates an Eigen::Isometry3f from a pose vector.
 * 
 * @param poseVector The pose vector [x, y, theta].
 * @return Eigen::Isometry3f The resulting isometry.
 */
Eigen::Isometry3f createIsometryFromPose(const Eigen::Vector3f& poseVector) {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    float theta = poseVector.coeff(2);
    Eigen::Matrix3f rotationMatrix;
    rotationMatrix << std::cos(theta), -std::sin(theta), 0,
                      std::sin(theta),  std::cos(theta), 0,
                      0,                0,               1;
    pose.linear() = rotationMatrix;
    // Create a Vector3f for translation: use the first two components from poseVector and set z to 0.
    pose.translation() = Eigen::Vector3f(poseVector.coeff(0), poseVector.coeff(1), 0.0f);
    return pose;
}


/**
 * @brief Creates an Eigen::Isometry3f from rotation matrix and translation vector.
 * 
 * @param rotationMatrix The rotation matrix (3x3).
 * @param translationVector The translation vector (3x1).
 * @return Eigen::Isometry3f The resulting isometry.
 */
Eigen::Isometry3f createIsometryFromRt(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& translationVector) {
    Eigen::Isometry3f iso = Eigen::Isometry3f::Identity();
    iso.linear() = rotationMatrix;
    iso.translation() = translationVector;
    return iso;
}

/**
 * @brief Converts a cv::Mat to an Eigen::Vector3f.
 * 
 * @param inputMat The input matrix (3x1, CV_32F).
 * @return Eigen::Vector3f The resulting Eigen vector.
 */
Eigen::Vector3f cvToEigenVector(const cv::Mat& inputMat) {
    if (inputMat.rows != 3 || inputMat.cols != 1) {
        throw std::invalid_argument("Input cv::Mat must be a 3x1 vector of type CV_32F.");
    }
    Eigen::Vector3f eigenVec;
    for (int i = 0; i < 3; ++i) {
        eigenVec.coeffRef(i) = inputMat.at<float>(i, 0);
    }
    return eigenVec;
}

/**
 * @brief Converts a cv::Mat to an Eigen::Matrix3f.
 * 
 * @param inputMat The input matrix (3x3, CV_32F).
 * @return Eigen::Matrix3f The resulting Eigen matrix.
 */
Eigen::Matrix3f cvToEigenMatrix(const cv::Mat& inputMat) {
    if (inputMat.rows != 3 || inputMat.cols != 3) {
        throw std::invalid_argument("Input cv::Mat must be a 3x3 matrix.");
    }
    Eigen::Matrix3f eigenMat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigenMat(i, j) = inputMat.at<float>(i, j);
        }
    }
    return eigenMat;
}

/**
 * @brief Prints an Eigen::Isometry3f to the output stream.
 * 
 * @param os The output stream.
 * @param iso The isometry to print.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& os, const Eigen::Isometry3f& iso) {
    // Print the translation vector
    os << "Translation: [" 
       << iso.translation().x() << ", "
       << iso.translation().y() << ", "
       << iso.translation().z() << "]\n";

    // Print the rotation matrix
    os << "Rotation:\n" << iso.rotation() << "\n";

    return os;
}

/**
 * @brief Prints a vector of Eigen::Isometry3f.
 * 
 * @param os The output stream.
 * @param vec The vector to print.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& os, const std::vector<Eigen::Isometry3f>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        os << "Isometry " << i << ":\n" << vec[i] << "\n";
    }
    return os;
}

/**
 * @brief Prints a vector of floats.
 * 
 * @param os The output stream.
 * @param vec The vector to print.
 * @return std::ostream& The output stream.
 */
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", "; // Add a comma between elements
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Counts the number of correct matches based on real IDs.
 * 
 * @param matched_ids_real The matched ID pairs.
 * @return int The number of correct matches.
 */
int count_correct_matches(const std::vector<Eigen::Vector2i>& matched_ids_real) {
    int correct_matches = 0;

    for (const auto& match : matched_ids_real) {
        if (match(0) == match(1)) {
            correct_matches++;
        }
    }

    return correct_matches;
}

/**
 * @brief Converts an Eigen::Isometry3f to a cv::Mat (4x4).
 * 
 * @param isometry The isometry to convert.
 * @return cv::Mat The resulting cv::Mat.
 */
cv::Mat isometry3fToCvMat(const Eigen::Isometry3f& isometry) {
    cv::Mat pose_matrix(4, 4, CV_32F, cv::Scalar(0));
    
    // Copy rotation matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            pose_matrix.at<float>(i, j) = isometry.linear()(i, j);
        }
    }
    
    // Copy translation vector
    for (int i = 0; i < 3; ++i) {
        pose_matrix.at<float>(i, 3) = isometry.translation()(i);
    }
    
    // Set bottom row
    pose_matrix.at<float>(3, 3) = 1.0f;
    
    return pose_matrix;
}

/**
 * @brief Computes the scale factor between reconstructed and ground truth points.
 * 
 * @param points_reconstructed The reconstructed points.
 * @param points_ground_truth The ground truth points.
 * @return float The computed scale factor.
 */
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

/**
 * @brief Computes the scale factor between two sets of cv::Mat points.
 * 
 * @param points1 The first set of points.
 * @param points2 The second set of points.
 * @return float The computed scale factor.
 */
float compute_scale(const cv::Mat& points1, const cv::Mat& points2) {
    if (points1.rows != points2.rows || points1.cols != points2.cols) {
        throw std::runtime_error("Point clouds must have same dimensions");
    }
    
    float total_scale = 0.0f;
    int valid_points = 0;
    
    for (int i = 0; i < points1.rows; ++i) {
        cv::Vec3f p1(points1.at<float>(i, 0), points1.at<float>(i, 1), points1.at<float>(i, 2));
        cv::Vec3f p2(points2.at<float>(i, 0), points2.at<float>(i, 1), points2.at<float>(i, 2));
        
        float dist1 = cv::norm(p1);
        float dist2 = cv::norm(p2);
        
        if (dist1 > 0 && dist2 > 0) {
            total_scale += dist2 / dist1;
            valid_points++;
        }
    }
    
    return (valid_points > 0) ? (total_scale / valid_points) : 1.0f;
}

/**
 * @brief Extracts world points based on correspondences.
 * 
 * @param world_points The complete world points matrix.
 * @param correspondences The matched correspondences.
 * @return cv::Mat The extracted world points.
 */
cv::Mat extract_world_points(const cv::Mat& world_points, 
                             const std::vector<Eigen::Vector2i>& correspondences) {
    
    cv::Mat extracted_points(correspondences.size(), world_points.cols, CV_32F);
    
    for (size_t i = 0; i < correspondences.size(); ++i) {
        int index = correspondences[i](0);  // Assuming first element is world_id
        if (index >= 0 && index < world_points.rows) {
            world_points.row(index).copyTo(extracted_points.row(i));
        } else {
            std::cerr << "Warning: World point index out of bounds (" << index << ")" << std::endl;
        }
    }
    
    return extracted_points;
}

/**
 * @brief Rescales a set of points by a given scale factor.
 * 
 * @param points The input points matrix.
 * @param scale The scale factor.
 * @return cv::Mat The rescaled points matrix.
 */
cv::Mat rescale_points(const cv::Mat& points, float scale) {
    cv::Mat scaled_points = points.clone();
    scaled_points *= scale;
    return scaled_points;
}

/**
 * @brief Filters correspondences into common and new correspondences.
 * 
 * @param all_correspondences All past correspondences.
 * @param common_correspondences The filtered common correspondences.
 * @param new_correspondences The filtered new correspondences.
 * @param previous_world_points The existing world points matrix.
 * @param new_world_points The matrix to store new world points.
 */
void filter_correspondences(
    const std::vector<std::vector<Eigen::Vector2i>>& all_correspondences, 
    std::vector<Eigen::Vector2i>& common_correspondences, 
    std::vector<Eigen::Vector2i>& new_correspondences,
    const cv::Mat& previous_world_points,
    cv::Mat& new_world_points
) {
    if (all_correspondences.size() < 2) {
        std::cerr << "Not enough correspondences to filter." << std::endl;
        return;
    }

    // Get the latest and previous correspondences from the list
    const std::vector<Eigen::Vector2i>& latest_correspondences = all_correspondences.back();
    const std::vector<Eigen::Vector2i>& previous_correspondences = all_correspondences[all_correspondences.size() - 2];

    std::cout << "Previous correspondences points: " << previous_correspondences.size() << std::endl;
    std::cout << "Latest correspondences points: " << latest_correspondences.size() << std::endl;
    std::cout << "World points size: " << previous_world_points.rows << std::endl;

    // Iterate through the latest correspondences to find common and new matches
    for (const auto& latest_correspondence : latest_correspondences) {
        // Search for a correspondence in previous_correspondences where previous id_real matches current id_meas
        auto el = std::find_if(previous_correspondences.begin(), previous_correspondences.end(),
                            [&latest_correspondence](const Eigen::Vector2i& vec) {
                                return vec.coeff(1) == latest_correspondence.coeff(0);
                            });

        if (el != previous_correspondences.end()) {
            // Common correspondence found
            common_correspondences.push_back(*el);

            int idx = static_cast<int>(std::distance(previous_correspondences.begin(), el));

            // Ensure idx is within bounds
            if (idx >= 0 && idx < previous_world_points.rows) {
                new_world_points.push_back(previous_world_points.row(idx));
                // std::cout << "(" << idx << ") <-- (" << (*el)(0) << ", " << (*el)(1) << ") | "
                //           << previous_world_points.row(idx) << " <-- ("
                //           << latest_correspondence(0) << ", " << latest_correspondence(1) << ") | "
                //           << new_world_points.row(new_world_points.rows - 1) << std::endl;
            } else {
                std::cerr << "Warning: Index out of bounds for previous_world_points at index " << idx << std::endl;
            }
        } else {
            // New correspondence found
            new_correspondences.push_back(latest_correspondence);
        }
    }
}

/**
 * @brief Filters matches based on new correspondences.
 * 
 * @param img_points1 The first set of image points.
 * @param img_points2 The second set of image points.
 * @param correspondences The new correspondences.
 * @param filtered_img_points1 The filtered first set of image points.
 * @param filtered_img_points2 The filtered second set of image points.
 */
void filter_matches(const std::vector<Data_Point>& img_points1,
                    const std::vector<Data_Point>& img_points2,
                    const std::vector<Eigen::Vector2i>& correspondences,
                    std::vector<Data_Point>& filtered_img_points1,
                    std::vector<Data_Point>& filtered_img_points2) {

    std::cout << "img_points1 size: " << img_points1.size() << std::endl;
    std::cout << "img_points2 size: " << img_points2.size() << std::endl;
    std::cout << "correspondences size: " << correspondences.size() << std::endl;

    filtered_img_points1.clear();
    filtered_img_points2.clear();

    if (correspondences.empty()) {
        return;
    }

    filtered_img_points1.reserve(correspondences.size());
    filtered_img_points2.reserve(correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i) {
        const int idx1 = correspondences[i].coeff(0);
        const int idx2 = correspondences[i].coeff(1);


        // Check that indices are non-negative and within bounds
        if (idx1 < 0 || idx2 < 0 ||
            idx1 >= static_cast<int>(img_points1.size()) ||
            idx2 >= static_cast<int>(img_points2.size())) {
            std::cerr << "Warning: Index out of bounds for correspondence at index " << i << std::endl;
            continue;
            }
        filtered_img_points1.push_back(img_points1[idx1]);
        filtered_img_points2.push_back(img_points2[idx2]);
    }

    std::cout << "filtered_img_points1 size: " << filtered_img_points1.size() << std::endl;
    std::cout << "filtered_img_points2 size: " << filtered_img_points2.size() << std::endl;
}

/**
 * @brief Converts a vector of Eigen::Vector2i to pr::IntPairVector by swapping elements.
 * 
 * @param eigenVector The input vector of Eigen::Vector2i.
 * @return pr::IntPairVector The resulting pr::IntPairVector.
 */
pr::IntPairVector eigenToIntPairVector(const std::vector<Eigen::Vector2i>& eigenVector) {
    pr::IntPairVector intPairVector;
    intPairVector.reserve(eigenVector.size());
    for (const Eigen::Vector2i& vec : eigenVector) {
        intPairVector.push_back(std::make_pair(
            static_cast<int>(vec.coeff(1)),
            static_cast<int>(vec.coeff(0))
        ));
    }
    return intPairVector;
}


/**
 * @brief Creates an Eigen::Isometry3f from a pose vector.
 * 
 * @param pose The pose vector [x, y, theta].
 * @return Eigen::Isometry3f The resulting isometry.
 */
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


/**
 * @brief Computes the relative motion error between estimated and ground truth poses.
 * 
 * @param rel_T The relative estimated pose.
 * @param rel_GT The relative ground truth pose.
 */
void computeRelativeMotionError(const Eigen::Isometry3f& rel_T, const Eigen::Isometry3f& rel_GT) {
    // Compute the relative error transformation: error_T = inv(rel_T) * rel_GT
    Eigen::Isometry3f error_T = rel_T.inverse() * rel_GT;

    // Rotation error: trace(I - error_T.rotation())
    Eigen::Matrix3f error_R = error_T.rotation(); // Extract the 3x3 rotation matrix
    float rotation_error = (Eigen::Matrix3f::Identity() - error_R).trace();

    // Translation error: norm of translation parts, compute scale ratio
    Eigen::Vector3f trans_rel_T = rel_T.translation(); // Extract translation from rel_T
    Eigen::Vector3f trans_rel_GT = rel_GT.translation(); // Extract translation from rel_GT
    float norm_rel_T = trans_rel_T.norm();
    float norm_rel_GT = trans_rel_GT.norm();
    float scale_ratio = norm_rel_T / norm_rel_GT;

    // Output the results
    std::cout << "Rotation Error (trace): " << rotation_error << std::endl;
    std::cout << "Scale Ratio: " << scale_ratio << std::endl;
}

/**
 * @brief Visualizes matched points between two images.
 * 
 * @param matched_points1 The first set of matched points.
 * @param matched_points2 The second set of matched points.
 */
void visualizeMatches(const cv::Mat& matched_points1, const cv::Mat& matched_points2) {
    // Ensure both matched_points1 and matched_points2 have the same number of rows
    if (matched_points1.rows != matched_points2.rows) {
        std::cerr << "Error: Number of matched points do not match." << std::endl;
        return;
    }

    std::vector<cv::Point2f> keypoints1, keypoints2;

    // Convert 2D points to KeyPoint format
    for (int i = 0; i < matched_points1.rows; ++i) {
        keypoints1.emplace_back(cv::Point2f(matched_points1.at<float>(i, 0), matched_points1.at<float>(i, 1)));
        keypoints2.emplace_back(cv::Point2f(matched_points2.at<float>(i, 0), matched_points2.at<float>(i, 1)));
    }

    // Create synthetic images
    cv::Mat img1 = cv::Mat::zeros(480, 640, CV_8UC3); 
    cv::Mat img2 = cv::Mat::zeros(480, 640, CV_8UC3); 

    // Draw keypoints
    for (const auto& kp : keypoints1) {
        cv::circle(img1, kp, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto& kp : keypoints2) {
        cv::circle(img2, kp, 2, cv::Scalar(0, 0, 255), -1);
    }

    std::vector<cv::DMatch> matches; 
    for (int i = 0; i < keypoints1.size(); ++i) {
        matches.emplace_back(cv::DMatch(i, i, 0));
    }

    // Visualize matches
    cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
}

/**
 * @brief Runs one round of the PICP solver to estimate the camera pose.
 * 
 * @param last_pose_estimate The last estimated pose (camera_to_world).
 * @param pr_cam The PICP Camera object.
 * @param world_points The current world points.
 * @param image_points The current image points.
 * @param correspondences The correspondences between world and image points.
 * @return Eigen::Isometry3f The updated estimated pose (camera_to_world).
 */
Eigen::Isometry3f oneRound(Eigen::Isometry3f last_pose_estimate, 
                          pr::Camera& pr_cam, 
                          const pr::Vector3fVector& world_points, 
                          const pr::Vector2fVector& image_points, 
                          const pr::IntPairVector& correspondences /* add the world in camera pose isometry */ ) {

    int maxIterations = 100;
    double convergenceThreshold = 0.00001;
    double prevError = std::numeric_limits<double>::max();
    double currentError;

    pr::PICPSolver solver;
    pr_cam.setWorldInCameraPose(last_pose_estimate);
    solver.init(pr_cam, world_points, image_points);

    for (int i = 0; i < maxIterations; i++) {
        solver.oneRound(correspondences, false);
        currentError = solver.chiInliers();
        
        if (std::abs(prevError - currentError) < convergenceThreshold) {
            break;
        }
        prevError = currentError;
    }
    // std::cout << "error (inliers): " << currentError << " (outliers): " << solver.chiOutliers() << std::endl;
    return solver.camera().worldInCameraPose();
}

/**
 * @brief Prints a vector of Eigen::Vector3f.
 * 
 * @param vec The vector to print.
 */
void printVector3fVector(const Vector3fVector& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        const Eigen::Vector3f& v = vec[i];
        std::cout << "Vector " << i << ": (" 
                  << v[0] << ", " 
                  << v[1] << ", " 
                  << v[2] << ")" 
                  << std::endl;
    }
}

/**
 * @brief Prints a vector of Eigen::Vector2i.
 * 
 * @param vec The vector to print.
 */
void printVector2i(const std::vector<Eigen::Vector2i>& vec) {
    std::cout << "Vector contents:" << std::endl;
    for (const Eigen::Vector2i& v : vec) {
        std::cout << "[" << v.coeff(0) << ", " << v.coeff(1) << "]" << std::endl;
    }
}

/**
 * @brief Creates a plot of the ground truth and estimated trajectories.
 * 
 * @param gt_poses The ground truth poses.
 * @param est_poses The estimated poses.
 * @param title The title of the plot window.
 */
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

/**
 * @brief Maps a vector of objects to a vector of a specific member.
 * 
 * @tparam T The type of the objects in the input vector.
 * @tparam U The type of the member to extract.
 * @param vec The input vector of objects.
 * @param member The member to extract.
 * @return std::vector<U> The resulting vector of extracted members.
 */
template <typename T, typename U>
std::vector<U> vec_map(const std::vector<T>& vec, U T::*member) {
    std::vector<U> result;
    result.reserve(vec.size());
    for (const auto& obj : vec) {
        result.push_back(obj.*member);
    }
    return result;
}
