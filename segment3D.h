#ifndef I3D_LINE3D_PP_SEGMENT3D_H_
#define I3D_LINE3D_PP_SEGMENT3D_H_

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

// external
#include "eigen3/Eigen/Eigen"

// std
#include <list>

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

        Segment3D(const Eigen::Vector3d P1,
                  const Eigen::Vector3d P2)
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
        float distance_Point2Line(const Eigen::Vector3d P) const
        {
            Eigen::Vector3d hlp_pt = P1_ + (dir_ * ((P - P1_).transpose()) * dir_);
            return (hlp_pt-P).norm();
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
    };

    //------------------------------------------------------------------------------
    // one reconstructed 3D line cluster
    class LineCluster3D
    {
    public:
        LineCluster3D(){}
        LineCluster3D(L3DPP::Segment3D seg3D,
                      L3DPP::Segment2D correspondingSeg2D,
                      std::list<L3DPP::Segment2D> residuals) :
            seg3D_(seg3D), correspondingSeg2D_(correspondingSeg2D),
            residuals_(residuals){}

        // data access
        L3DPP::Segment3D seg3D() const {return seg3D_;}
        L3DPP::Segment2D correspondingSeg2D() const {return correspondingSeg2D_;}
        std::list<L3DPP::Segment2D>* residuals(){return &residuals_;}
        size_t size(){return residuals_.size();}

        // update 3D line (after bundling)
        void update3Dline(L3DPP::Segment3D seg3D){
            seg3D_ = seg3D;
        }

    private:
        L3DPP::Segment3D seg3D_;
        L3DPP::Segment2D correspondingSeg2D_;
        std::list<L3DPP::Segment2D> residuals_;
    };

    //------------------------------------------------------------------------------
    // final 3D line result, with collinear 3D segments
    struct FinalLine3D
    {
        std::list<L3DPP::Segment3D> collinear3Dsegments_;
        L3DPP::LineCluster3D underlyingCluster_;
    };
}

#endif //I3D_LINE3D_PP_SEGMENT3D_H_
