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
    cv::Mat K = (cv::Mat_<double>(3, 3) << 180, 0, 320,
                                            0, 180, 240,
                                            0,   0,   1);

    cv::Mat transform = (cv::Mat_<double>(4, 4) << 0, 0, 1, 0.2,
                                                -1, 0, 0, 0,
                                                0,-1, 0, 0,
                                                0, 0, 0, 1);

    double z_near = 0;
    double z_far = 5;
    int width = 640;
    int height = 480;           
};

struct Point3D {
    cv::Mat coord_3D = cv::Mat_<double>(1, 3);
    vector<double> appearance = vector<double>(10);
};

struct Measurement {
    int seq;
    vector<double> gt_pose = vector<double>(3);
    vector<double> odometry_pose = vector<double>(3);
    vector<int> ids_meas;
    vector<int> ids_real;
    cv::Mat points2D = cv::Mat_<double>(0, 2);
    cv::Mat appearances = cv::Mat_<double>(0, 10);
};

double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    double sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void match_points(const vector<Measurement>& measurements, const int idx1, const int idx2, cv::Mat& matches1, cv::Mat& matches2, cv::Mat& matched_ids) {
    Measurement m1 = measurements[idx1];
    Measurement m2 = measurements[idx2];
    double threshold = 0.7;

    std::cout << "Finding best matches for each point..." << endl;

    // compute the minimum euclidean distance between each point in m1 and each point in m2 based on the appearance vector
    // and add the pair of points to the matches matrix and the ids of points to the matched_ids matrix    
    for (int i = 0; i < m1.appearances.rows; i++) {
        double min_distance = std::numeric_limits<double>::max();
        int min_idx = -1;
        for (int j = 0; j < m2.appearances.rows; j++) {
            double distance = euclidean_distance(m1.appearances.row(i), m2.appearances.row(j));
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
            cv::Mat match1 = cv::Mat_<double>(1, 2);
            cv::Mat match2 = cv::Mat_<double>(1, 2);
            match1.at<double>(0) = m1.points2D.at<double>(i, 0);
            match1.at<double>(1) = m1.points2D.at<double>(i, 1);
            match2.at<double>(0) = m2.points2D.at<double>(min_idx, 0);
            match2.at<double>(1) = m2.points2D.at<double>(min_idx, 1);
            matches1.push_back(match1);
            matches2.push_back(match2);
            cv::Mat match_ids = cv::Mat_<int>(1, 2);
            match_ids.at<int>(0) = m1.ids_meas[i];
            match_ids.at<int>(1) = m2.ids_meas[min_idx];
            matched_ids.push_back(match_ids);
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

    if (t_hat.at<double>(2) < 0) {
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
        point.coord_3D.at<double>(0) = stof(tokens[1]);
        point.coord_3D.at<double>(1) = stof(tokens[2]);
        point.coord_3D.at<double>(2) = stof(tokens[3]);
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

    cv::Mat identity = cv::Mat::eye(3, 4, CV_64F);
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

int visualize_coordinates(cv::Mat point_matrix, cv::Mat matched_points1, cv::Mat matched_points2) {

    // Visualizing coordinates
    vector<cv::Point3f> cloud;
    for (int i = 0; i < point_matrix.rows; i++ ){
        cv::Point3d p(  point_matrix.at<double>( i , 0 ),
                        point_matrix.at<double>( i , 1 ),
                        point_matrix.at<double>( i , 2 ) );
        cloud.push_back( p );
    }
    /*
    // visualize 2D vector points matched_points1 and matched_points2 with different colors
    vector<cv::Point3f> points1;
    for (int i = 0; i < matched_points1; i++) {
        cv::Point3d p(  matched_points1.at<double>( i , 0 ),
                        matched_points1.at<double>( i , 1 ),
                        0.0 );
        points1.push_back( p );
    }
    */

    // Create a window
    cv::viz::Viz3d window("3D Points");

    // Create a WCloud object and set its properties
    cv::viz::WCloud cloud_widget( cloud, cv::viz::Color::white() );
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 10);
    // Add the WCloud object to the window
    window.showWidget("cloud", cloud_widget);

    cv::viz::WCoordinateSystem cs(1.0);
    window.showWidget("CoordinateSystem", cs);

    //cv::viz::Color white(255, 255, 255); // RGB values for white
    //window.setBackgroundColor(white);

    //cv::Vec3d from(1.0, 1.0, 10.0); // camera position
    //cv::Vec3d to(0., 0., 0.); // look at the center of the point cloud
    //cv::Vec3d up(0.0, 0.0, 1.0); // up direction
    //window.setViewerPose(cv::Affine3d(cv::Matx33d::eye(), from));

    // Show the visualization
    window.spin();
    return 0;  
}