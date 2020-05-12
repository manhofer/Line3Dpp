#ifndef I3D_LINE3D_PP_SEGMENT3D_H_
#define I3D_LINE3D_PP_SEGMENT3D_H_

/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// external
#include "eigen3/Eigen/Eigen"

// std
#include <list>
#include <set>

// internal
#include "commons.h"

/**
 * Line3D++ - Segment3D Class
 * ====================
 * Defines a 3D line segment.
 * ====================
 * Author: M.Hofer, 2016
 */

namespace L3DPP
{
    //------------------------------------------------------------------------------
    // one 3D line segment
    class Segment3D
    {
    public:
        Segment3D()
        {
            P1_ = Eigen::Vector3d(0,0,0);
            P2_ = Eigen::Vector3d(0,0,0);
            dir_ = Eigen::Vector3d(0,0,0);
            length_ = 0.0f;
            valid_ = false;
        }

        Segment3D(const Eigen::Vector3d& P1,
                  const Eigen::Vector3d& P2)
        {
            length_ = (P1-P2).norm();
            if(length_ > L3D_EPS)
            {
                P1_ = P1;
                P2_ = P2;
                dir_ = (P2-P1).normalized();
                valid_ = true;
            }
            else
            {
                P1_ = Eigen::Vector3d(0,0,0);
                P2_ = Eigen::Vector3d(0,0,0);
                dir_ = Eigen::Vector3d(0,0,0);
                length_ = 0.0f;
                valid_ = false;
            }
        }

        // distance point to line
        float distance_Point2Line(const Eigen::Vector3d& P) const
        {
            Eigen::Vector3d hlp_pt = P1_ + (dir_ * ((P - P1_).transpose()) * dir_);
            return (hlp_pt-P).norm();
        }

        // translate points
        void translate(const Eigen::Vector3d& t)
        {
            P1_ += t;
            P2_ += t;
        }

        // data access
        Eigen::Vector3d P1() const {return P1_;}
        Eigen::Vector3d P2() const {return P2_;}
        Eigen::Vector3d dir() const {return dir_;}
        float length() const {return length_;}
        bool valid() const {return valid_;}

    private:
        Eigen::Vector3d P1_;
        Eigen::Vector3d P2_;
        Eigen::Vector3d dir_;
        float length_;
        bool valid_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("length_", length_);
            ar & boost::serialization::make_nvp("valid_", valid_);

            ar & boost::serialization::make_nvp("P1_x", P1_.x());
            ar & boost::serialization::make_nvp("P1_y", P1_.y());
            ar & boost::serialization::make_nvp("P1_z", P1_.z());

            ar & boost::serialization::make_nvp("P2_x", P2_.x());
            ar & boost::serialization::make_nvp("P2_y", P2_.y());
            ar & boost::serialization::make_nvp("P2_z", P2_.z());

            ar & boost::serialization::make_nvp("dir_x", dir_.x());
            ar & boost::serialization::make_nvp("dir_y", dir_.y());
            ar & boost::serialization::make_nvp("dir_z", dir_.z());
        }
    };

    //------------------------------------------------------------------------------
    // one reconstructed 3D line cluster
    class LineCluster3D
    {
    public:
        LineCluster3D(){}
        LineCluster3D(const L3DPP::Segment3D& seg3D,
                      const std::list<L3DPP::Segment2D>& residuals,
                      const unsigned int ref_view) :
            seg3D_(seg3D), residuals_(residuals),
            reference_view_(ref_view){}

        // data access
        L3DPP::Segment3D seg3D() const {return seg3D_;}
        const std::list<L3DPP::Segment2D>* residuals() const {return &residuals_;}
        size_t size() const {return residuals_.size();}
        unsigned int reference_view() const {return reference_view_;}

        // update 3D line (after bundling)
        void update3Dline(const L3DPP::Segment3D& seg3D){
            seg3D_ = seg3D;
        }

        // translate
        void translate(const Eigen::Vector3d& t)
        {
            seg3D_.translate(t);
        }

    private:
        L3DPP::Segment3D seg3D_;
        std::list<L3DPP::Segment2D> residuals_;
        unsigned int reference_view_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("seg3D_", seg3D_);
            ar & boost::serialization::make_nvp("residuals_", residuals_);
            ar & boost::serialization::make_nvp("reference_view_", reference_view_);
        }
    };

    //------------------------------------------------------------------------------
    // final 3D line result, with collinear 3D segments
    struct FinalLine3D
    {
        std::list<L3DPP::Segment3D> collinear3Dsegments_;
        L3DPP::LineCluster3D underlyingCluster_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("collinear3Dsegments_", collinear3Dsegments_);
            ar & boost::serialization::make_nvp("underlyingCluster_", underlyingCluster_);
        }
    };

    //------------------------------------------------------------------------------
    // final 3D point (from SfM)
    struct FinalPoint3D
    {
        Eigen::Vector3d P3D_;
        // specifies in which cameras this point is visible (i.e. has a residual)
        std::set<unsigned int> visibility_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("visibility_", visibility_);
            ar & boost::serialization::make_nvp("P3D_x", P3D_.x());
            ar & boost::serialization::make_nvp("P3D_y", P3D_.y());
            ar & boost::serialization::make_nvp("P3D_z", P3D_.z());
        }
    };
}

#endif //I3D_LINE3D_PP_SEGMENT3D_H_
