#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <opencv2/viz.hpp>

using namespace std;
using std::vector;
using std::pair;

struct Camera {
    cv::Mat K = (cv::Mat_<float>(3, 3) << 180, 0, 320,
                                            0, 180, 240,
                                            0,   0,   1);

    /*
    cv::Mat transform = (cv::Mat_<float>(4, 4) << 0, 0, 1, 0.2,
                                                -1, 0, 0, 0,
                                                0,-1, 0, 0,
                                                0, 0, 0, 1);
    */  

    float z_near = 0;
    float z_far = 5;
    int width = 640;
    int height = 480;           
};

struct Point3D {
    cv::Mat coord_3D = cv::Mat_<float>(1, 3);
    vector<float> appearance = vector<float>(10);
};

struct Measurement {
    int seq;
    vector<float> gt_pose = vector<float>(3);
    vector<float> odometry_pose = vector<float>(3);
    vector<int> ids_meas;
    vector<int> ids_real;
    cv::Mat points2D = cv::Mat_<float>(0, 2);
    cv::Mat appearances = cv::Mat_<float>(0, 10);
};

float euclidean_distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void match_points(const vector<Measurement>& measurements, const int idx1, const int idx2, cv::Mat& matches1, cv::Mat& matches2, cv::Mat& matched_ids) {
    Measurement m1 = measurements[idx1];
    Measurement m2 = measurements[idx2];
    float threshold = 0.7;

    if (idx1 < 0 || idx1 >= measurements.size() || idx2 < 0 || idx2 >= measurements.size()) {
        std::cerr << "Invalid indices for measurements." << std::endl;
        return;
    }

    std::cout << "Finding best matches for each point..." << endl;

    // compute the minimum euclidean distance between each point in m1 and each point in m2 based on the appearance vector
    // and add the pair of points to the matches matrix and the ids of points to the matched_ids matrix    
    for (int i = 0; i < m1.appearances.rows; i++) {
        float min_distance = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j = 0; j < m2.appearances.rows; j++) {
            float distance = euclidean_distance(m1.appearances.row(i), m2.appearances.row(j));
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = j;
            }
        }
        // control if indices go out of bounds on both sides
        if (min_idx == -1) {
            std::cout << "min_idx is -1" << endl;
        }
        
        // control if the minimum distance is less than the threshold
        if (min_distance < threshold) {
            cv::Mat match1 = cv::Mat_<float>(1, 2);
            cv::Mat match2 = cv::Mat_<float>(1, 2);
            
            match1.at<float>(0, 0) = m1.points2D.at<float>(i, 0);
            match1.at<float>(0, 1) = m1.points2D.at<float>(i, 1);
            match2.at<float>(0, 0) = m2.points2D.at<float>(min_idx, 0);
            match2.at<float>(0, 1) = m2.points2D.at<float>(min_idx, 1);
            matches1.push_back(match1);
            matches2.push_back(match2);
            cv::Mat match_ids = cv::Mat_<int>(1, 2);
            match_ids.at<int>(0) = m1.ids_meas[i];
            match_ids.at<int>(1) = m2.ids_meas[min_idx];
            matched_ids.push_back(match_ids);
                        
            // cout << "Matched Point Pair: (" << i << ", " << min_idx << ")" << endl;

        }
    }
    
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

void decompose_fund_matrix(const cv::Mat& F, const cv::Mat& K, vector<cv::Point2f> points1, vector<cv::Point2f> points2, cv::Mat& R, cv::Mat& t)
{
    cv::Mat E, R1, R2, t_hat;

    cout << "F size: " << F.size() << ", type: " << F.type() << endl;
    cout << "K size: " << K.size() << ", type: " << K.type() << endl;
    cv:: Mat K_new;
    K.convertTo(K_new, F.type());
    //F.convertTo(F, K.type());
    cout << "K_new size: " << K_new.size() << ", type: " << K_new.type() << endl;

    E = K_new.t() * F * K_new;

    bool success = recoverPose(E, points1, points2, K_new, R1, R2, t_hat); 
    if (!success) {
        std::cout << "Failed to recover pose." << std::endl;
    }

    if (t_hat.at<float>(2) < 0) {
        R = R2;
        t = -t_hat;
    } else {
        R = R1;
        t = t_hat;
    }
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
            meas.points2D.push_back(stof(tokens[3]));
            meas.points2D.push_back(stof(tokens[4]));

            // add tokens from 5 to 14 to appearances
            for (int i=5; i<15; i++) {
                meas.appearances.push_back(stof(tokens[i]));
            }

        } else {
            std::cerr << "Invalid line in file " << filename << ": " << line << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    // reshape meas.points2D so that each row is a point
    meas.points2D = meas.points2D.reshape(1, meas.points2D.size().height/2);
    // reshape meas.appearances so that each row is an appearance
    meas.appearances = meas.appearances.reshape(1, meas.appearances.size().height/10);

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

vector<Point3D> extract_world(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        exit(EXIT_FAILURE);
    }
    Point3D point;
    vector<Point3D> points;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> tokens = split(line, " ");
        point.coord_3D.at<float>(0) = stof(tokens[1]);
        point.coord_3D.at<float>(1) = stof(tokens[2]);
        point.coord_3D.at<float>(2) = stof(tokens[3]);
        for (int i=0; i<10; i++) {
            point.appearance[i] = stof(tokens[i+4]);
        }
        points.push_back(point);
    }
    return points;
}

cv::Mat triangulate(cv::Mat& R, cv::Mat& t, cv::Mat& K, cv::Mat& matches1 , cv::Mat& matches2) {

    // print function description
    cout << "Triangulating points..." << endl;

    cv::Mat identity = cv::Mat::eye(3, 4, CV_32F);
    cv::Mat P1 = K * identity;   
    cv::Mat P2;
    cv::hconcat(R, t, P2);
    cv::Mat points3D, points4D;
    
    // triangulate points and convert to 3D points
    cv::triangulatePoints(P1, P2, matches1.t(), matches2.t(), points4D);
    
    // convert to homogeneous coordinates
    cv::convertPointsFromHomogeneous(points4D.t(), points3D);

    // print triangulated points
    cout << points4D.t() << endl;

    // print number of found points
    cout << "Found " << points3D.rows << " points" << endl;
    
   return points3D;
}

void visualize_3d_points(const cv::Mat& point_matrix) {
    // Create a window
    cv::viz::Viz3d window("3D Points");

    // Create a vector to hold the 3D points
    std::vector<cv::Point3f> cloud;

    // Fill the cloud vector with points from the input matrix
    for (int i = 0; i < point_matrix.rows; i++) {
        cv::Point3f p(point_matrix.at<float>(i, 0),
                      point_matrix.at<float>(i, 1),
                      point_matrix.at<float>(i, 2));
        cloud.push_back(p);
    }

    // Create a WCloud object and set its properties
    cv::viz::WCloud cloud_widget(cloud, cv::viz::Color::white());
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 5);

    // Add the WCloud object to the window
    window.showWidget("cloud", cloud_widget);

    // Create a coordinate system
    cv::viz::WCoordinateSystem cs(1.0);
    window.showWidget("CoordinateSystem", cs);

    // Show the visualization
    window.spin();
}

void visualize_matched_points(const cv::Mat& matches1, const cv::Mat& matches2) {
    // Create an empty canvas to draw the matched points
    cv::Mat canvas(500, 500, CV_8UC3, cv::Scalar(255, 255, 255)); // White canvas

    // Loop through the matched points and draw them on the canvas
    for (int i = 0; i < matches1.rows; i++) {
        cv::Point2f pt1(matches1.at<float>(i, 0), matches1.at<float>(i, 1));
        cv::Point2f pt2(matches2.at<float>(i, 0), matches2.at<float>(i, 1));

        // Draw a line connecting the matched points
        cv::line(canvas, pt1, pt2, cv::Scalar(0, 0, 0), 1);  // Black line
        cv::circle(canvas, pt1, 3, cv::Scalar(0, 0, 255), -1); // Red circle at pt1
        cv::circle(canvas, pt2, 3, cv::Scalar(0, 0, 255), -1); // Red circle at pt2
    }

    // Display the canvas with matched points and lines
    cv::imshow("Matched Points on 2D Plane", canvas);
    cv::waitKey(0);
}

void visualize_2d_matched_points(const cv::Mat& matches1, const cv::Mat& matches2) {
    // Create a Viz3d window
    cv::viz::Viz3d window("2D Points in 3D");

    // Create a vector to hold the 3D points
    std::vector<cv::Point3d> cloud;

    // Fill the cloud vector with points from matches1 and matches2
    for (int i = 0; i < matches1.rows; i++) {
        cv::Point2d p1(matches1.at<float>(i, 0), matches1.at<float>(i, 1));
        cv::Point2d p2(matches2.at<float>(i, 0), matches2.at<float>(i, 1));
        
        // Create 3D points with Z-coordinate set to 0
        cv::Point3d p1_3d(p1.x, p1.y, 0.0);
        cv::Point3d p2_3d(p2.x, p2.y, 0.0);
        
        // Add the 3D points to the cloud
        cloud.push_back(p1_3d);
        cloud.push_back(p2_3d);
    }

    // Create a WCloud object and set its properties
    cv::viz::WCloud cloud_widget(cloud, cv::viz::Color::white());
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 10);


    // Add the cloud and coordinate system to the window
    window.showWidget("Cloud", cloud_widget);

    // Create an affine transformation to set the camera pose
    cv::Affine3d pose = cv::viz::makeTransformToGlobal(cv::Vec3d(0.0, 0.0, 3.0), cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, -1.0, 0.0));
    window.setViewerPose(pose);

    // Show the visualization
    window.spin();
}

string size(cv::Mat m) {
    return "[" + std::to_string(m.rows) + " x " + std::to_string(m.cols) + "]";
}