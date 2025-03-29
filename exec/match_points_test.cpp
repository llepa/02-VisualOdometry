#include <Eigen/Core>

#include "../src/my_utilities.h"
#include "../src/defs.h"
#include "../src/data_point.h"

#include <string>
#include <vector>


int main(int argc, char** argv) {
    int n_meas = 121;   // Total number of measurements (frames)
    std::string meas_path_prefix = "./data/meas-";    // Path prefix for measurement files
    std::string world_filename = "./data/world.dat";  // Ground truth world points file

    // Load measurements and ground truth world points
    std::vector<Measurement> measurements = load_and_initialize_data(meas_path_prefix, n_meas);
    std::vector<World_Point> world_points_gt = load_world_points(world_filename);

    for (int i = 0; i < n_meas - 1; i++) {
        std::vector<Data_Point> points1 = measurements[i].data_points;
        std::vector<Data_Point> points2 = measurements[i + 1].data_points;
        IntPairVector correspondences;

        std::vector<std::pair<Data_Point, Data_Point>> matches;
        match_points(points1, points2, matches, correspondences);

        int n_matches = 0;
        for (int j = 0; j < matches.size(); j++) {
            auto& match = matches[j];
            int idx1 = correspondences[j].first;
            int idx2 = correspondences[j].second;
            if (match.first.id_real == match.second.id_real
                && points1[idx1].id_real == points2[idx2].id_real){
                n_matches++;
            }
        }
        std::cout << "Iteration: " << i << ": matches: " << n_matches << " / " << matches.size() << std::endl;
    }

}