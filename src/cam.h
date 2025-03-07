#ifndef CAM_H
#define CAM_H

#include "data_point.h"  // Now Data_Point is defined here
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp> // For eigen2cv and cv2eigen

/**
 * @class Cam
 * @brief A class representing a camera object that can compute essential matrices, recover poses, 
 *        and triangulate points.
 */
class Cam {
public:
    /**
     * @brief Constructs a Cam object and initializes camera parameters.
     */
    Cam();

    /**
     * @brief Computes the essential matrix from matched points and recovers the pose (R and t).
     *        The computed rotation and translation are stored internally.
     *
     * @param mask Output mask indicating inlier matches after RANSAC.
     */
    void computeEssentialAndRecoverPose(const std::vector<std::pair<Data_Point, Data_Point>>& matches,
                                        cv::Mat& mask);

    /**
     * @brief Triangulates points given two projection matrices and matched keypoints.
     * 
     * @param T1 Projection matrix [R1|t1] for the first view (4x4, CV_32F).
     * @param T2 Projection matrix [R2|t2] for the second view (4x4, CV_32F).
     * @param matches
     * @param points3D Output 3D points in homogeneous coordinates (Nx3, CV_32F).
     */
    void triangulatePoints(const cv::Mat& T1,
                            const cv::Mat& T2,
                            std::vector<std::pair<Data_Point, Data_Point>>& matches,
                            cv::Mat& points3D);

    /**
     * @brief Projects 3D points into the image given rotation and translation.
     * 
     * @param R Rotation matrix (3x3, CV_32F).
     * @param t Translation vector (3x1, CV_32F).
     * @param points3D 3D points (Nx3, CV_32F) to project.
     * @return cv::Mat 2D projections (Nx2, CV_32F).
     */
    cv::Mat projectPoints(const cv::Mat& R, 
                          const cv::Mat& t, 
                          const cv::Mat& points3D);

    /**
     * @brief Visualizes a set of 3D points. This is a placeholder function and can be implemented 
     *        as needed (e.g., using CV Viz, Pangolin, or another visualization library).
     * 
     * @param point_matrix Nx3 CV_32F matrix containing 3D points.
     */
    void visualizePoints(const cv::Mat& point_matrix);

    /**
     * @brief Returns the camera intrinsic matrix as an Eigen::Matrix3f.
     * 
     * @return Eigen::Matrix3f The camera intrinsic matrix.
     */
    Eigen::Matrix3f getEigenCamera();

    /**
     * @brief Normalizes the translation vector. This is useful to handle scale ambiguities 
     *        in monocular setups.
     * 
     * @param t The translation vector (3x1, CV_32F) to be normalized.
     */
    void normalize_translation(cv::Mat& t);

    /**
     * @brief Returns the current rotation matrix (3x3, CV_32F) recovered by the last call to 
     *        computeEssentialAndRecoverPose.
     * 
     * @return cv::Mat The rotation matrix.
     */
    cv::Mat getRotationMatrix() const { return R_; }

    /**
     * @brief Returns the current translation vector (3x1, CV_32F) recovered by the last call to 
     *        computeEssentialAndRecoverPose.
     * 
     * @return cv::Mat The translation vector.
     */
    cv::Mat getTranslationVector() const { return t_; }

private:
    cv::Mat K_cv;          ///< Camera intrinsic matrix in OpenCV format.
    Eigen::Matrix3f K_eig; ///< Camera intrinsic matrix in Eigen format.
    float z_near;          ///< Near clipping distance (if used for visualization).
    float z_far;           ///< Far clipping distance (if used for visualization).
    int width;             ///< Image width.
    int height;            ///< Image height.

    cv::Mat R_;            ///< Rotation matrix recovered from essential matrix (3x3, CV_32F).
    cv::Mat t_;            ///< Translation vector recovered from essential matrix (3x1, CV_32F).

    /**
     * @brief Computes the projection matrices P1 and P2 for triangulation.
     * 
     * @param R1 Rotation of the first camera (3x3, CV_32F).
     * @param t1 Translation of the first camera (3x1, CV_32F).
     * @param R2 Rotation of the second camera (3x3, CV_32F).
     * @param t2 Translation of the second camera (3x1, CV_32F).
     * @param P1 Output projection matrix for the first camera (3x4, CV_32F).
     * @param P2 Output projection matrix for the second camera (3x4, CV_32F).
     */
    void _computeProjectionMatrices(const cv::Mat& R1, 
                                    const cv::Mat& t1,
                                    const cv::Mat& R2, 
                                    const cv::Mat& t2,
                                    cv::Mat& P1, 
                                    cv::Mat& P2);
};

#endif // CAM_H
