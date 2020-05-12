#ifndef I3D_LINE3D_PP_COMMONS_H_
#define I3D_LINE3D_PP_COMMONS_H_

/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// check libs
#include "configLIBS.h"

// internal
#include "serialization.h"

// external
#include <queue>
#include <stdlib.h>

// windows fix
#if _WIN32
#define M_PI   3.14159265358979323846264338327950288
#define M_PI_2 1.57079632679489661923132169163975144
#endif


/**
 * Line3D++ - Constants
 * ====================
 * Default parameters, etc.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    // feature detection
    #define L3D_DEF_MAX_IMG_WIDTH -1
    #define L3D_DEF_MIN_IMG_WIDTH 800
    #define L3D_DEF_MIN_LINE_LENGTH_FACTOR 0.005f
    #define L3D_DEF_MAX_NUM_SEGMENTS 3000
    #define L3D_DEF_LOAD_AND_STORE_SEGMENTS true

    // collinearity
    #define L3D_DEF_COLLINEARITY_T -1.0f

    // matching
    #define L3D_DEF_MATCHING_NEIGHBORS 10
    #define L3D_DEF_EPIPOLAR_OVERLAP 0.25f
    #define L3D_DEF_KNN 10
    #define L3D_DEF_SCORING_POS_REGULARIZER 2.5f
    #define L3D_DEF_SCORING_ANG_REGULARIZER 10.0f
    #define L3D_DEF_CHECK_MATCH_ORIENTATION true

    // scoring
    #define L3D_DEF_MIN_SIMILARITY_3D 0.50f
    #define L3D_DEF_MIN_BEST_SCORE_3D 0.75f
    #define L3D_DEF_MIN_BEST_SCORE_PERC 0.10f

    // replicator dynamics diffusion
    #define L3D_DEF_PERFORM_RDD false
    #define L3D_DEF_RDD_MAX_ITER 10

    // clustering
    #define L3D_DEF_MIN_AFFINITY 0.50f
    #define L3D_DEF_MIN_VISIBILITY_T 3

    // Plane3D: superpixels
    #define P3D_DEF_MAX_IMG_WIDTH 720
    #define P3D_DEF_MIN_IMG_WIDTH 640
    #define P3D_DEF_NUM_SUPERPIXELS 500
    #define P3D_DEF_SUPERPIXEL_W 40

    // Plane3D: hypothesis selection/generation
    #define P3D_DEF_HYP_NEIGHBORS 10
    #define P3D_DEF_ANG_REGULARIZER 10.0f
    #define P3D_DEF_POS_REGULARIZER 2.5f

    // optimization
#ifdef L3DPP_CERES
    #define L3D_DEF_USE_CERES true
#else
    #define L3D_DEF_USE_CERES false
#endif //L3DPP_CERES
    #define L3D_DEF_CERES_MAX_ITER 250

    // display
    #define L3D_DISP_CAMS 4
    #define L3D_DISP_LINES 5
    #define L3D_DISP_MATCHES 9

    #define L3D_EPS 1e-12
    #define L3D_PI_1_2 1.5707963267948966f
    #define L3D_PI_1_4 0.785398163f
    #define L3D_PI_3_4 2.35619449f
    #define L3D_PI_1_32 0.098174771f
    #define L3D_PI_31_32 3.043417886f

    //------------------------------------------------------------------------------
    // 2D segment (sortable)
    class Segment2D
    {
    public:
        Segment2D() : camID_(0), segID_(0){}
        Segment2D(unsigned int camID,
                     unsigned int segID) :
            camID_(camID), segID_(segID){}
        inline unsigned int camID() const {return camID_;}
        inline unsigned int segID() const {return segID_;}

        inline bool operator== (const Segment2D& rhs) const {return ((camID_ == rhs.camID_) && (segID_ == rhs.segID_));}
        inline bool operator< (const Segment2D& rhs) const {
            return ((camID_ < rhs.camID_) || (camID_ == rhs.camID_ && segID_ < rhs.segID_));
        }
        inline bool operator!= (const Segment2D& rhs) const {return !((*this) == rhs);}
    private:
        unsigned int camID_;
        unsigned int segID_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("camID_", camID_);
            ar & boost::serialization::make_nvp("segID_", segID_);
        }
    };

    //------------------------------------------------------------------------------
    // 2D segment coords (with length)
    struct SegmentData2D
    {
        float p1x_;
        float p1y_;
        float p2x_;
        float p2y_;
        float length_;
    };

    // comparator
    class SegmentData2D_comparison
    {
    public:
        SegmentData2D_comparison(){}
        bool operator() (const L3DPP::SegmentData2D& lhs, const L3DPP::SegmentData2D& rhs) const
        {
            return lhs.length_ < rhs.length_;
        }
    };

    // queue
    typedef std::priority_queue<L3DPP::SegmentData2D,std::vector<L3DPP::SegmentData2D>,L3DPP::SegmentData2D_comparison> lines2D_sorted_by_length;

    //------------------------------------------------------------------------------
    // visual neighbors
    struct VisualNeighbor
    {
        unsigned int camID_;
        float score_;
        float axisAngle_;
        float distance_score_;
    };

    // visual neighbor comparators
    static bool sortVisualNeighborsByScore(const L3DPP::VisualNeighbor lhs, const L3DPP::VisualNeighbor rhs)
    {
        return lhs.score_ > rhs.score_;
    }

    static bool sortVisualNeighborsByAngle(const L3DPP::VisualNeighbor lhs, const L3DPP::VisualNeighbor rhs)
    {
        return lhs.axisAngle_ > rhs.axisAngle_;
    }

    static bool sortVisualNeighborsByDistScore(const L3DPP::VisualNeighbor lhs, const L3DPP::VisualNeighbor rhs)
    {
        return lhs.distance_score_ > rhs.distance_score_;
    }

    //------------------------------------------------------------------------------
    // potential match
    struct Match
    {
        // correspondence
        unsigned int src_camID_;
        unsigned int src_segID_;
        unsigned int tgt_camID_;
        unsigned int tgt_segID_;

        // scores
        float overlap_score_;
        float score3D_;

        // depths
        float depth_p1_;
        float depth_p2_;
        float depth_q1_;
        float depth_q2_;
    };

    // sorting functions
    static bool sortMatchesByIDs(const Match m1, const Match m2)
    {
        if(m1.tgt_camID_ < m2.tgt_camID_)
            return true;
        else if(m1.tgt_camID_ == m2.tgt_camID_ && m1.tgt_segID_ < m2.tgt_segID_)
            return true;
        else
            return false;
    }

    // match comparator (for kNN matching)
    class Match_kNN
    {
    public:
        Match_kNN(){}
        bool operator() (const Match& lhs, const Match& rhs) const
        {
            return lhs.overlap_score_ < rhs.overlap_score_;
        }
    };

    // queue
    typedef std::priority_queue<L3DPP::Match,std::vector<L3DPP::Match>,L3DPP::Match_kNN> pairwise_matches;

    //------------------------------------------------------------------------------
    // poinr on clustered 3D line
    struct PointOn3DLine
    {
        size_t lineID_;
        size_t pointID_;
        size_t camID_;
        float distToBorder_;
    };

    static bool sortPointsOn3DLine(const L3DPP::PointOn3DLine p1, const L3DPP::PointOn3DLine p2)
    {
        return p1.distToBorder_ < p2.distToBorder_;
    }
}

#endif //I3D_LINE3D_PP_COMMONS_H_
