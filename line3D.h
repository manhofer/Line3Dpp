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
 * Reference:
 * "Efficient 3D Scene Abstraction Using Line Segments"
 * Manuel Hofer, Michael Maurer, Horst Bischof,
 * Computer Vision and Image Understanding (CVIU), 2016.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    class Line3D
    {
    public:
        // Line3D++ constructor
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // output_folder            - folder where the temp directory will be created
        // load_segments            - if true  -> detected 2D line segments will be serialized to hard drive
        //                                        and reloaded when the dataset is processed again
        //                            if false -> the line segments are redetected everytime
        // max_img_width            - maximum width (or height, for portrait images) to which images are resized
        //                            for line segment detection (coordinates will be upscaled afterwards!).
        //                            if set to -1, the images are not resized
        // max_line_segments        - maximum number of 2D line segments per image (sorted by length)
        // neighbors_by_worldpoints - if true  -> matching neighbors (images) are derived from the common worldpoints
        //                            if false -> an explicit list of matching neighbors has to be provided
        //                            (--> see void addImage(...))
        // use_GPU                  - uses the GPU for processing whenever possible (highly recommended, requires CUDA!)
        Line3D(const std::string& output_folder,
               const bool load_segments=L3D_DEF_LOAD_AND_STORE_SEGMENTS,
               const int max_img_width=L3D_DEF_MAX_IMG_WIDTH,
               const unsigned int max_line_segments=L3D_DEF_MAX_NUM_SEGMENTS,
               const bool neighbors_by_worldpoints=true,
               const bool use_GPU=true);
        ~Line3D();

        // void addImage(...): add a new image to the system [multithreading safe]
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // camID            - unique ID of the image
        // image            - the image itself (CV_8U or CV_8UC3 supported)
        // K                - camera intrinsics (3x3)
        // R                - camera rotation (3x3)
        // t                - camera translation (3x1) [camera model: point2D = K [R | t] point3D]
        // median_depth     - median 3D worldpoint depth for this camera (i.e. median
        //                    Euclidean distance of the worldpoints to the camera center)
        // wps_or_neighbors - a list with the IDs of the
        //                    (a) worldpoints seen by this camera                --> if neighbors_by_worldpoints=true (see constructor)
        //                    (b) images with which this image should be matched --> if neighbors_by_worldpoints=false
        // line_segments    - list with the 2D line segments for this image. if it is empty (default) the line segments
        //                    will be detected by the LSD algorithm automatically
        void addImage(const unsigned int camID, cv::Mat& image,
                      const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& t, const float median_depth,
                      const std::list<unsigned int>& wps_or_neighbors,
                      const std::vector<cv::Vec4f>& line_segments=std::vector<cv::Vec4f>());

        // void undistortImage(...): undistorts an image based on given distortion coefficients
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // inImg             - distorted image (input)
        // outImg            - undistorted image (output)
        // radial_coeffs     - up to three radial distortion coefficients (set unused to zero!)
        // tangential_coeffs - up tp two tangential distortion coefficients (set unused to zero!)
        // K                 - camera intrinsics (3x3)
        static void undistortImage(const cv::Mat& inImg, cv::Mat& outImg,
                                   const Eigen::Vector3d& radial_coeffs,
                                   const Eigen::Vector2d& tangential_coeffs,
                                   const Eigen::Matrix3d& K);

        // void matchImages(...): matches 2D line segments between images
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // sigma_position             - spatial regularizer (for scoring and clustering)
        //                              if > 0 -> in pixels (regularizer derived from image space and unprojected into 3D space) [scale invariant]
        //                              if < 0 -> in "meters" (regularizer directly defined in world coordinates) [not scale invariant]
        //                              the second method is recommended when the scale is known!
        // sigma_angle                - angular regularizer (for scoring and clustering)
        //                              defined in degrees (not radiants!)
        // num_neighbors              - number of neighboring images with which each image is matched
        // epipolar_overlap           - minimum overlap of a line segment with the epipolar beam of another segment,
        //                              to be considered a potential match (in [0,1])
        // kNN                        - k-nearest-neighbor matching
        //                              if > 0  -> keep only the k matches with the highest epipolar overlap (per image)
        //                              if <= 0 -> keep all matches that fulfill the epipolar_overlap
        // const_regularization_depth - if positive (and sigma_position is in "meters"), this depth is where
        //                              an uncertainty of 'sigma_position' is allowed (e.g. use 5.0 when you want to
        //                              initialize sigma_p 5 meters in front of the camera)
        void matchImages(const float sigma_position=L3D_DEF_SCORING_POS_REGULARIZER,
                         const float sigma_angle=L3D_DEF_SCORING_ANG_REGULARIZER,
                         const unsigned int num_neighbors=L3D_DEF_MATCHING_NEIGHBORS,
                         const float epipolar_overlap=L3D_DEF_EPIPOLAR_OVERLAP,
                         const int kNN=L3D_DEF_KNN,
                         const float const_regularization_depth=-1.0f);

        // void reconstruct3Dlines(...): reconstruct a line-based 3D model (after matching)
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // visibility_t      - minimum number of different cameras from which clustered 2D segments must originate,
        //                     such that the resulting 3D line is considered to be valid
        // perform_diffusion - perform Replicator Dynamics Diffusion [Donoser, BMVC'13] before
        //                     segment clustering
        // collinearity_t    - threshold (in pixels) for segments from one image to be considered potentially collinear
        //                     if <= 0 -> collinearity not considered (default)
        // use_CERES         - 3D lines are optimized (bundled) using the Ceres-Solver (recommended!)
        // max_iter_CERES    - maximum number of iterations for Ceres
        void reconstruct3Dlines(const unsigned int visibility_t=L3D_DEF_MIN_VISIBILITY_T,
                                const bool perform_diffusion=L3D_DEF_PERFORM_RDD,
                                const float collinearity_t=L3D_DEF_COLLINEARITY_T,
                                const bool use_CERES=L3D_DEF_USE_CERES,
                                const unsigned int max_iter_CERES=L3D_DEF_CERES_MAX_ITER);

        // void get3Dlines(...): returns the current 3D model
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // result - list of reconstructed 3D lines (see "segment3D.h")
        void get3Dlines(std::vector<L3DPP::FinalLine3D>& result);

        // void saveResultAs*(...): saves current 3D model in different ways
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // output_folder - folder where to place the result
        //
        // Note: see README.md for a description of the output formats!
        void saveResultAsSTL(const std::string& output_folder);
        void saveResultAsOBJ(const std::string& output_folder);
        void save3DLinesAsTXT(const std::string& output_folder);
        void save3DLinesAsBIN(const std::string& output_folder);

        // Eigen::Vector4f getSegmentCoords2D(...): provides access to the 2D segment coordinates
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // seg2D - desired 2D segment (camID and segmentID)
        //
        // camID - camera ID of desired 2D line segment
        // segID - segment ID of desired 2D line segment
        Eigen::Vector4f getSegmentCoords2D(const L3DPP::Segment2D& seg2D);
        Eigen::Vector4f getSegmentCoords2D(const unsigned int camID,
                                           const unsigned int segID);

        // size_t numImages(...): returns the number of images that have been added (addImage(...))
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // none
        size_t numImages(){return views_.size();}

        // L3DPP::DataArray<float4>* detectLineSegments(...): detects line segments in an image
        // -------------------------------------
        // PARAMETERS:
        // -------------------------------------
        // camID - ID of the given view (for re-loading only)
        // image - the corresponding image (cv::Mat, CV_8U or CV_8UC3)
        L3DPP::DataArray<float4>* detectLineSegments(const unsigned int camID, const cv::Mat& image);

        // --------------------------------------------------
        // helper functions (needed in specific executables):
        // --------------------------------------------------

        // rotation matrix from roll, pitch and yaw (rodriguez)
        static Eigen::Matrix3d rotationFromRPY(const double roll, const double pitch,
                                               const double yaw);

        // rotation matrix from a quaternion
        static Eigen::Matrix3d rotationFromQ(const double Qw, const double Qx,
                                             const double Qy, const double Qz);

        // create an output filename based on the current parameter settings
        std::string createOutputFilename();

        // decompose P matrix to K, R and t
        static void decomposeProjectionMatrix(const Eigen::MatrixXd P_in,
                                              Eigen::Matrix3d& K_out,
                                              Eigen::Matrix3d& R_out,
                                              Eigen::Vector3d& t_out);

    private:
        // process worldpoint list
        void processWPlist(const unsigned int camID, const std::list<unsigned int>& wps);

        // store visual neighbors directly
        void setVisualNeighbors(const unsigned int camID, const std::list<unsigned int>& neighbors);

        // find visual neighbors
        void findVisualNeighborsFromWPs(const unsigned int camID);

        // initialize/cleanup src data on/from GPU
        void initSrcDataGPU(const unsigned int src);
        void removeSrcDataGPU(const unsigned int src);

        // compute matches between images
        void computeMatches();
        void matchingCPU(const unsigned int src, const unsigned int tgt,
                         const Eigen::Matrix3d& F);
        void matchingGPU(const unsigned int src, const unsigned int tgt,
                         const Eigen::Matrix3d& F);

        // get fundamental matrix
        Eigen::Matrix3d getFundamentalMatrix(L3DPP::View* src, L3DPP::View* tgt);

        // check if a given point x is inside a line segment p1,p2 (point must be on the line!)
        bool pointOnSegment(const Eigen::Vector3d& x, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& p2);

        // compute segment overlap (input: 4 collinear points)
        float mutualOverlap(const std::vector<Eigen::Vector3d>& collinear_points);

        // compute endpoint depths for a line segment, based on a match
        Eigen::Vector2d triangulationDepths(const unsigned int src_camID, const Eigen::Vector3d& p1,
                                            const Eigen::Vector3d& p2, const unsigned int tgt_camID,
                                            const Eigen::Vector3d& line_q1, const Eigen::Vector3d& line_q2);

        // sort matches for each source segment
        void sortMatches(const unsigned int src);

        // score matches
        void scoringCPU(const unsigned int src, float& valid_f);
        void scoringGPU(const unsigned int src, float& valid_f);

        // similarity between two matches/segments
        float similarityForScoring(const L3DPP::Match& m1, const L3DPP::Match& m2,
                                   const L3DPP::Segment3D& seg3D1,
                                   const float reg1, const float reg2);
        float similarity(const L3DPP::Segment2D& seg1, const L3DPP::Segment2D& seg2,
                         const bool truncate);
        float similarity(const L3DPP::Segment3D& s1, const L3DPP::Match& m1,
                         const L3DPP::Segment2D& seg2, const bool truncate);

        // angle between two segments (in degrees!)
        float angleBetweenSeg3D(const L3DPP::Segment3D& s1, const L3DPP::Segment3D& s2,
                                const bool undirected=true);

        // unproject match to 3D segment
        L3DPP::Segment3D unprojectMatch(const L3DPP::Match& m, const bool src=true);

        // check match orientation (angle between optical axis and 3D segment)
        void checkMatchOrientation(const unsigned int src);

        // store new matches for other image as well
        void storeInverseMatches(const unsigned int src);

        // filter out invalid matches
        void filterMatches(const unsigned int src);

        // find collinear 2D segments (per image)
        void findCollinearSegments();

        // computing affinity matrix
        void computingAffinityMatrix();
        bool unused(const L3DPP::Segment2D& seg1, const L3DPP::Segment2D& seg2);

        // get a local ID for clustering (=row index in A matrix)
        int getLocalID(const L3DPP::Segment2D& seg);

        // perform replicator dynamics diffusion on A
        void performRDD();

        // cluster affinity matrix
        void clusterSegments();

        // compute final 3D segments from clusters
        void computeFinal3Dsegments();

        // filter tiny segments
        void filterTinySegments();

        // get 3D line from clustered 3D lines
        L3DPP::LineCluster3D get3DlineFromCluster(const std::list<L3DPP::Segment2D>& cluster);

        // project 2D segment onto 3D line
        L3DPP::Segment3D project2DsegmentOnto3Dline(const L3DPP::Segment2D& seg2D,
                                                    const L3DPP::Segment3D& seg3D,
                                                    bool& success);

        // compute collinear segments on cluster
        std::list<L3DPP::Segment3D> findCollinearSegments(const L3DPP::LineCluster3D& cluster);

        // optimize (bundle) 3D line clusters
        void optimizeClusters();

        // converts Eigen::Matrix to DataArray<float>
        void eigen2dataArray(L3DPP::DataArray<float>* &DA, const Eigen::MatrixXd& M);

        // saves a temporary 3D model as .stl
        void saveTempResultAsSTL(const std::string& output_folder,
                                 const std::string& suffix,
                                 const std::vector<L3DPP::Segment3D>& lines3D);

        // translate/untranslate views and 3D models (for better numerical stability)
        void translate();
        void untranslate();
        void performTranslation(const Eigen::Vector3d t);

        // basic params
        std::string data_folder_;
        std::string prefix_;
        std::string prefix_err_;
        std::string prefix_wng_;
        bool useGPU_;
        boost::mutex display_text_mutex_;

        // line segment detection
        unsigned int max_line_segments_;
        int max_image_width_;
        bool load_segments_;
        float collinearity_t_;

        // view data
        unsigned int num_lines_total_;
        boost::mutex view_mutex_;
        boost::mutex view_reserve_mutex_;
        std::vector<unsigned int> view_order_;
        std::map<unsigned int,L3DPP::View*> views_;
        std::set<unsigned int> views_reserved_;
        std::vector<float> views_avg_depths_;
        float med_scene_depth_;
        float med_scene_depth_lines_;
        Eigen::Vector3d translation_;

        // neighbors
        bool neighbors_by_worldpoints_;
        std::map<unsigned int,std::list<unsigned int> > worldpoints2views_;
        std::map<unsigned int,std::list<unsigned int> > views2worldpoints_;
        std::map<unsigned int,std::list<unsigned int> > fixed_visual_neighbors_;
        std::map<unsigned int,unsigned int> num_worldpoints_;
        std::map<unsigned int,std::set<unsigned int> > visual_neighbors_;

        // matches
        unsigned int num_neighbors_;
        float epipolar_overlap_;
        int kNN_;
        boost::mutex match_mutex_;
        boost::mutex scoring_mutex_;
        std::map<unsigned int,std::set<unsigned int> > matched_;
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
        float const_regularization_depth_;

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
        std::map<L3DPP::Segment2D,std::set<L3DPP::Segment2D> > used_;
    };
}

#endif //I3D_LINE3D_PP_LINE3D_H_
