#ifndef I3D_LINE3D_PP_LINE3D_H_
#define I3D_LINE3D_PP_LINE3D_H_

/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// check libs
#include "configLIBS.h"

// std
#include <map>
#include <queue>
#include <iostream>
#include <iomanip>

// external
#include "eigen3/Eigen/Eigen"
#include "boost/filesystem.hpp"
#include "boost/thread/mutex.hpp"

// OpenCV and LSD
#ifndef L3DPP_OPENCV3
#include "opencv/cv.h"
#include "lsd/lsd_opencv.hpp"
#else
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#endif //L3DPP_OPENCV3

// internal
#include "clustering.h"
#include "commons.h"
#include "view.h"
#include "dataArray.h"
#include "serialization.h"
#include "segment3D.h"
#include "cudawrapper.h"
#include "optimization.h"
#include "sparsematrix.h"

/**
 * Line3D++ - Base Class
 * ====================
 * Line-based Multi-view Stereo
 * Reference: [add thesis]
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    class Line3D
    {
    public:
        Line3D(const std::string output_folder,
               const bool load_segments=L3D_DEF_LOAD_AND_STORE_SEGMENTS,
               const unsigned int max_img_width=L3D_DEF_MAX_IMG_WIDTH,
               const unsigned int max_line_segments=L3D_DEF_MAX_NUM_SEGMENTS,
               const bool neighbors_by_worldpoints=true,
               const bool use_GPU=true);
        ~Line3D();

        // add image [multithreading safe]
        // (1) visual neighbors computed from common worldpoints
        // (2) visual neighbors explicitely given
        // --> depends on parameter "neighbors_by_worldpoints" in constructor!
        void addImage(const unsigned int camID, cv::Mat& image,
                      const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                      const Eigen::Vector3d t, const float median_depth,
                      std::list<unsigned int>& wps_or_neighbors,
                      std::vector<cv::Vec4f> line_segments=std::vector<cv::Vec4f>());

        // match images
        void matchImages(const float sigma_position=L3D_DEF_SCORING_POS_REGULARIZER,
                         const float sigma_angle=L3D_DEF_SCORING_ANG_REGULARIZER,
                         const unsigned int num_neighbors=L3D_DEF_MATCHING_NEIGHBORS,
                         const float epipolar_overlap=L3D_DEF_EPIPOLAR_OVERLAP,
                         const float min_baseline=L3D_DEF_MIN_BASELINE,
                         const int kNN=L3D_DEF_KNN);

        // reconstruct 3D model
        void reconstruct3Dlines(const unsigned int visibility_t=L3D_DEF_MIN_VISIBILITY_T,
                                const bool perform_diffusion=L3D_DEF_PERFORM_RDD,
                                const float collinearity_t=L3D_DEF_COLLINEARITY_T,
                                const bool use_CERES=L3D_DEF_USE_CERES,
                                const unsigned int max_iter_CERES=L3D_DEF_CERES_MAX_ITER);

        // get reconstructed 3D model
        void get3Dlines(std::vector<L3DPP::FinalLine3D>& result);

        // save result
        void saveResultAsSTL(const std::string output_folder);
        void saveResultAsOBJ(const std::string output_folder);
        void save3DLinesAsTXT(const std::string output_folder);

        // rotation matrix from roll, pitch and yaw (rodriguez)
        Eigen::Matrix3d rotationFromRPY(const double roll, const double pitch,
                                        const double yaw);

        // data access
        size_t numImages(){return views_.size();}

        // segment access
        Eigen::Vector4f getSegmentCoords2D(const L3DPP::Segment2D seg2D);
        Eigen::Vector4f getSegmentCoords2D(const unsigned int camID,
                                           const unsigned int segID);

        // create an output filename based on the parameter settings
        std::string createOutputFilename();

        // undistorts an image based on up to three radial distortion-
        // and two tangential distortion parameters
        void undistortImage(const cv::Mat& inImg, cv::Mat& outImg,
                            const Eigen::Vector3d radial_coeffs,
                            const Eigen::Vector2d tangential_coeffs,
                            const Eigen::Matrix3d& K);

    private:
        // process worldpoint list
        void processWPlist(const unsigned int camID, std::list<unsigned int>& wps);

        // store visual neighbors directly
        void setVisualNeighbors(const unsigned int camID, std::list<unsigned int>& neighbors);

        // detect/load line segments
        L3DPP::DataArray<float4>* detectLineSegments(const unsigned int camID, cv::Mat& image);

        // find visual neighbors
        void findVisualNeighborsFromWPs(const unsigned int camID);

        // initialize/cleanup src data on/from GPU
        void initSrcDataGPU(const unsigned int src);
        void removeSrcDataGPU(const unsigned int src);

        // compute matches between images
        void computeMatches();
        void matchingCPU(const unsigned int src, const unsigned int tgt,
                         Eigen::Matrix3d& F);
        void matchingGPU(const unsigned int src, const unsigned int tgt,
                         Eigen::Matrix3d& F);

        // get fundamental matrix
        Eigen::Matrix3d getFundamentalMatrix(L3DPP::View* src, L3DPP::View* tgt);

        // check if a given point x is inside a line segment p1,p2 (point must be on the line!)
        bool pointOnSegment(const Eigen::Vector3d x, const Eigen::Vector3d p1,
                            const Eigen::Vector3d p2);

        // compute segment overlap
        float mutualOverlap(std::vector<Eigen::Vector3d>& collinear_points);

        // compute endpoint depths for a line segment, based on a match
        Eigen::Vector2d triangulationDepths(const unsigned int src_camID, const Eigen::Vector3d p1,
                                            const Eigen::Vector3d p2, const unsigned int tgt_camID,
                                            const Eigen::Vector3d line_q1, const Eigen::Vector3d line_q2);

        // sort matches for each source segment
        void sortMatches(const unsigned int src);

        // score matches
        void scoringCPU(const unsigned int src, float& valid_f);
        void scoringGPU(const unsigned int src, float& valid_f);

        // similarity between two matches/segments
        float similarityForScoring(const L3DPP::Match m1, const L3DPP::Match m2,
                                   const float current_k1);
        float similarity(const L3DPP::Segment2D seg1, const L3DPP::Segment2D seg2,
                         const bool truncate);
        float similarity(const L3DPP::Segment3D s1, const L3DPP::Match m1,
                         const L3DPP::Segment2D seg2, const bool truncate);

        // angle between two segments (in degrees!)
        float angleBetweenSeg3D(const L3DPP::Segment3D s1, const L3DPP::Segment3D s2,
                                const bool undirected=true);

        // unproject match to 3D segment
        L3DPP::Segment3D unprojectMatch(const L3DPP::Match m, const bool src=true);

        // store new matches for other image as well
        void storeInverseMatches(const unsigned int src);

        // filter out invalid matches
        void filterMatches(const unsigned int src);

        // find collinear 2D segments (per image)
        void findCollinearSegments();

        // computing affinity matrix
        void computingAffinityMatrix();
        bool unused(const L3DPP::Segment2D seg1, const L3DPP::Segment2D seg2);

        // get a local ID for clustering (=row index in A matrix)
        int getLocalID(const L3DPP::Segment2D seg);

        // perform replicator dynamics diffusion on A
        void performRDD();

        // cluster affinity matrix
        void clusterSegments();

        // compute final 3D segments from clusters
        void computeFinal3Dsegments();

        // filter tiny segments
        void filterTinySegments();

        // get 3D line from clustered 3D lines
        L3DPP::LineCluster3D get3DlineFromCluster(std::list<L3DPP::Segment2D>& cluster);

        // project 2D segment onto 3D line
        L3DPP::Segment3D project2DsegmentOnto3Dline(const L3DPP::Segment2D seg2D,
                                                    const L3DPP::Segment3D seg3D,
                                                    bool& success);

        // compute collinear segments on cluster
        std::list<L3DPP::Segment3D> findCollinearSegments(L3DPP::LineCluster3D& cluster);

        // optimize (bundle) 3D line clusters
        void optimizeClusters();

        // converts Eigen::Matrix to DataArray<float>
        void eigen2dataArray(L3DPP::DataArray<float>* &DA, const Eigen::MatrixXd M);

        // basic params
        std::string data_folder_;
        std::string prefix_;
        std::string prefix_err_;
        std::string prefix_wng_;
        bool useGPU_;
        boost::mutex display_text_mutex_;

        // line segment detection
        unsigned int max_line_segments_;
        unsigned int max_image_width_;
        bool load_segments_;
        float collinearity_t_;

        // view data
        unsigned int num_lines_total_;
        boost::mutex view_mutex_;
        boost::mutex view_reserve_mutex_;
        std::vector<unsigned int> view_order_;
        std::map<unsigned int,L3DPP::View*> views_;
        std::map<unsigned int,bool> views_reserved_;

        // neighbors
        bool neighbors_by_worldpoints_;
        std::map<unsigned int,std::list<unsigned int> > worldpoints2views_;
        std::map<unsigned int,std::list<unsigned int> > views2worldpoints_;
        std::map<unsigned int,std::list<unsigned int> > fixed_visual_neighbors_;
        std::map<unsigned int,unsigned int> num_worldpoints_;
        std::map<unsigned int,std::map<unsigned int,bool> > visual_neighbors_;

        // matches
        unsigned int num_neighbors_;
        float min_baseline_;
        float epipolar_overlap_;
        int kNN_;
        boost::mutex match_mutex_;
        boost::mutex scoring_mutex_;
        std::map<unsigned int,std::map<unsigned int,bool> > matched_;
        std::map<unsigned int,std::map<unsigned int,Eigen::Matrix3d> > fundamentals_;
        std::map<unsigned int,std::vector<std::list<L3DPP::Match> > > matches_;
        std::map<unsigned int,unsigned int> num_matches_;
        std::map<unsigned int,bool> processed_;

        // scoring
        boost::mutex best_match_mutex_;
        std::vector<std::pair<L3DPP::Segment3D,L3DPP::Match> > estimated_position3D_;
        std::map<L3DPP::Segment2D,size_t> entry_map_;
        float sigma_p_;
        float sigma_a_;
        float two_sigA_sqr_;
        bool fixed3Dregularizer_;

        // reconstruction
        bool perform_RDD_;
        bool use_CERES_;
        unsigned int max_iter_CERES_;
        int localID_;
        unsigned int visibility_t_;
        boost::mutex aff_id_mutex_;
        boost::mutex aff_used_mutex_;
        boost::mutex aff_mat_mutex_;
        boost::mutex cluster_mutex_;
        std::list<L3DPP::CLEdge> A_;
        std::map<L3DPP::Segment2D,int> global2local_;
        std::map<int,L3DPP::Segment2D> local2global_;
        std::vector<L3DPP::LineCluster3D> clusters3D_;
        std::vector<L3DPP::FinalLine3D> lines3D_;
        std::map<L3DPP::Segment2D,std::map<L3DPP::Segment2D,bool> > used_;
    };
}

#endif //I3D_LINE3D_PP_LINE3D_H_
