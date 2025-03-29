#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "../src/my_utilities.h"
#include "../src/defs.h"
#include "../src/data_point.h"
#include "../src/cam.h"

#include <string>
#include <vector>


int main(int argc, char** argv) {

    int n_meas = 120;   // Total number of measurements (frames)
    std::string meas_path_prefix = "./data/meas-";    // Path prefix for measurement files
    std::string world_filename = "./data/world.dat";  // Ground truth world points file

    // Load measurements and ground truth world points
    std::vector<Measurement> measurements = load_and_initialize_data(meas_path_prefix, n_meas);
    // std::vector<World_Point> world_points_gt = load_world_points(world_filename);

    Cam cam = Cam();

    WorldPointVector world_points, triangulated_world_points;

    std::vector<Eigen::Isometry3f> poses, gt_poses;

    poses.push_back(Eigen::Isometry3f::Identity());
    gt_poses.push_back(Eigen::Isometry3f::Identity());

    for (int i = 0; i < 1; i++) {

        Eigen::Isometry3f gt_pose = augment_pose(measurements[i].gt_pose);
        gt_poses.push_back(gt_pose);

        std::cout << "\nIteration: "  << i << std::endl;

        std::vector<Data_Point> points1 = measurements[i].data_points;
        std::vector<Data_Point> points2 = measurements[i + 1].data_points;
        IntPairVector correspondences;

        std::vector<std::pair<Data_Point, Data_Point>> matches;
        match_points(points1, points2, matches, correspondences);

        // cv::Mat mask(matches.size(), 1, CV_8UC1);
        cv::Mat mask;
        cam.computeEssentialAndRecoverPose(matches, mask);

        Eigen::Isometry3f estimated_pose = cam.getPose();
        std::cout << "Pose: \n" << estimated_pose.matrix() << std::endl;

        cam.triangulatePoints(
            Eigen::Isometry3f::Identity(),
            estimated_pose,
            matches,
            world_points
        );

        std::cout << "\ncameraToImage: \n" << cam.cameraToImage().matrix() << std::endl;

        for (int j = 0; j < world_points.size(); j++) {
            World_Point wp = world_points[j];
            Eigen::Vector3f coordinates = Eigen::Vector3f(wp.coordinates.x, wp.coordinates.y, wp.coordinates.z);
            Eigen::Vector3f transformed_coordinates =
                cam.cameraToImage().linear().matrix() * coordinates
                + cam.cameraToImage().translation();
            std::cout << "\nReal ID world point: " << wp.id_real << std::endl;
            std::cout << transformed_coordinates << std::endl;
        }
    }
}