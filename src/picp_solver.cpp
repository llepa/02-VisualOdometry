#include "picp_solver.h"

#include <Eigen/Cholesky>
#include <iostream>

namespace pr {
  
  PICPSolver::PICPSolver(){
    _world_points=0;
    _reference_image_points=0;
    _damping=0.1;  // Reduced damping for smoother convergence
    _min_num_inliers=10;  // Set minimum number of inliers to ensure stability
    _num_inliers=0;
    _kernel_thereshold=100;  // Reduced kernel threshold from 1000 to 100 (~ 10 pixels)
  }

  void PICPSolver::init(const Camera& camera_,
                        const Vector3fVector& world_points,
                        const Vector2fVector& reference_image_points){
    _camera=camera_;
    _world_points=&world_points;
    _reference_image_points=&reference_image_points;
  }
  

  bool PICPSolver::errorAndJacobian(Eigen::Vector2f& error,
                                    Matrix2_6f& jacobian,
                                    const Eigen::Vector3f& world_point,
                                    const Eigen::Vector2f& reference_image_point){
    // compute the prediction
    Eigen::Vector2f predicted_image_point;
    bool is_good=_camera.projectPoint(predicted_image_point, world_point);
    if (! is_good)
      return false;
    error=predicted_image_point-reference_image_point;
    
    // compute the jacobian of the transformation
    Eigen::Vector3f camera_point=_camera.worldInCameraPose()*world_point;
    Matrix3_6f Jr=Eigen::Matrix<float, 3,6>::Zero();
    Jr.block<3,3>(0,0).setIdentity();
    Jr.block<3,3>(0,3)=skew(-camera_point);

    Eigen::Vector3f phom=_camera.cameraMatrix()*camera_point;
    float iz=1./phom.z();
    float iz2=iz*iz;
    // jacobian of the projection
    Matrix2_3f Jp;
    Jp << 
      iz, 0, -phom.x()*iz2,
      0, iz, -phom.y()*iz2;
      
    jacobian=Jp*_camera.cameraMatrix()*Jr;
    return true;
  }

  void PICPSolver::linearize(const IntPairVector& correspondences, bool keep_outliers){
    _H.setZero();
    _b.setZero();
    _num_inliers=0;
    _chi_inliers=0;
    _chi_outliers=0;
    
    // Count valid correspondences for better diagnostics
    int valid_correspondences = 0;
    
    for (const IntPair& correspondence: correspondences){
      Eigen::Vector2f e;
      Matrix2_6f J;
      int ref_idx=correspondence.first;
      int curr_idx=correspondence.second;
      
      // Make sure indices are within bounds
      if (ref_idx < 0 || ref_idx >= static_cast<int>(_reference_image_points->size()) || 
          curr_idx < 0 || curr_idx >= static_cast<int>(_world_points->size())) {
        continue;
      }
      
      bool inside=errorAndJacobian(e,
                                   J,
                                   (*_world_points)[curr_idx],
                                   (*_reference_image_points)[ref_idx]);
      if (!inside)
        continue;
        
      valid_correspondences++;
      
      float chi=e.dot(e);
      float lambda=1;
      bool is_inlier=true;
      
      if (chi>_kernel_thereshold){
        lambda=sqrt(_kernel_thereshold/chi);
        is_inlier=false;
        _chi_outliers+=chi;
      } else {
        _chi_inliers+=chi;
        _num_inliers++;
      }
      
      if (is_inlier || keep_outliers){
        _H+=J.transpose()*J*lambda;
        _b+=J.transpose()*e*lambda;
      }
    }
    
    if (valid_correspondences > 0) {
      std::cout << "Inliers/Total: " << _num_inliers << "/" << valid_correspondences 
                << " (" << (100.0f * _num_inliers / valid_correspondences) << "%)" << std::endl;
    }
  }

  bool PICPSolver::oneRound(const IntPairVector& correspondences, bool keep_outliers){
    using namespace std;
    linearize(correspondences, keep_outliers);
    
    // Add damping to the system - adaptive damping based on inlier ratio
    float effective_damping = _damping;
    if (!correspondences.empty()) {
      float inlier_ratio = static_cast<float>(_num_inliers) / correspondences.size();
      if (inlier_ratio < 0.5f) {
        // Increase damping if inlier ratio is low
        effective_damping *= (1.0f + (0.5f - inlier_ratio) * 2.0f);
      }
    }
    
    _H += Matrix6f::Identity() * effective_damping;
    
    if(_num_inliers < _min_num_inliers) {
      cerr << "Too few inliers (" << _num_inliers << " < " << _min_num_inliers << "), skipping optimization" << endl;
      return false;
    }
    
    // Compute a solution
    Vector6f dx = _H.ldlt().solve(-_b);
    
    // Check for numerical issues
    if (!isfinite(dx.sum())) {
      cerr << "Numerical issues detected in PICP solution" << endl;
      return false;
    }
    
    // Apply the solution
    _camera.setWorldInCameraPose(v2tEuler(dx) * _camera.worldInCameraPose());
    return true;
  }
}
