#ifndef I3D_LINE3D_PP_OPTIMIZATION_H_
#define I3D_LINE3D_PP_OPTIMIZATION_H_

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

// check if CERES is installed
#include "configLIBS.h"

#ifdef L3DPP_CERES

// ceres
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// external
#include <boost/thread.hpp>

// internal
#include "view.h"
#include "commons.h"

// std
#include <map>

/**
 * Line3D++ - Optimization Class
 * ====================
 * 3D line optimization (bundling)
 * using CERES.
 * ====================
 * Author: M.Hofer, 2016
 */

namespace L3DPP
{
    // data sizes
    const size_t LINE_SIZE = 6;
    const size_t INTRINSIC_SIZE = 4;
    const size_t CAM_PARAMETERS_SIZE = 6;
    const double LOSS_THRESHOLD = 2.0;

    // reprojection error for 3D line
    struct LineReprojectionError
    {
        LineReprojectionError(double observed_pt1_x, double observed_pt1_y,
                              double observed_pt2_x, double observed_pt2_y,
                              double observed_dir_l_x,
                              double observed_dir_l_y):
            observed_pt1_x_(observed_pt1_x),
            observed_pt1_y_(observed_pt1_y),
            observed_pt2_x_(observed_pt2_x),
            observed_pt2_y_(observed_pt2_y),
            observed_dir_l_x_(observed_dir_l_x),
            observed_dir_l_y_(observed_dir_l_y)
        {}

        template <typename T>
        bool operator()(const T* const camera,
                        const T* const line,
                        const T* const intrinsic,
                        T* residuals) const
        {
            // Translate into camera coordinate system
            T Q1[3];
            T Q2[3];

            Q1[0] = line[0]; Q1[1] = line[1]; Q1[2] = line[2];
            Q2[0] = line[3]; Q2[1] = line[4]; Q2[2] = line[5];

            T q1[3],q2[3];
            ceres::AngleAxisRotatePoint(camera, Q1, q1);
            ceres::AngleAxisRotatePoint(camera, Q2, q2);

            q1[0] += camera[3];
            q1[1] += camera[4];
            q1[2] += camera[5];

            q2[0] += camera[3];
            q2[1] += camera[4];
            q2[2] += camera[5];

            const T& px    = intrinsic[0];
            const T& py    = intrinsic[1];
            const T& fx    = intrinsic[2];
            const T& fy    = intrinsic[3];

            T x1n = (1.0 * q1[0] + 0.0 * q1[2]) / q1[2];
            T y1n = (1.0 * q1[1] + 0.0 * q1[2]) / q1[2];

            T x2n = (1.0 * q2[0] + 0.0 * q2[2]) / q2[2];
            T y2n = (1.0 * q2[1] + 0.0 * q2[2]) / q2[2];

            T xd1 = x1n;
            T yd1 = y1n;

            T xd2 = x2n;
            T yd2 = y2n;

            // projection function
            q1[0] = xd1;
            q1[1] = yd1;

            q2[0] = xd2;
            q2[1] = yd2;

            // projection function
            q1[0] = (fx * q1[0] + px);
            q1[1] = (fy * q1[1] + py);

            q2[0] = (fx * q2[0] + px);
            q2[1] = (fy * q2[1] + py);

            // compute infinite line (cross product)
            T l[3];
            l[0] = q1[1] - q2[1];
            l[1] = -(q1[0] - q2[0]);
            l[2] = q1[0]*q2[1] - q1[1]*q2[0];

            // normalize line
            T len_sqr = l[0]*l[0]+l[1]*l[1]+l[2]*l[2];
            T len_sqr1 = l[0]*l[0]+l[1]*l[1];

            if(len_sqr < 1e-12 || len_sqr1 < 1e-12)
            {
                residuals[0] = T(0.0);
                residuals[1] = T(0.0);
                return false;
            }

            // point distance
            T dx = q2[0]-q1[0];
            T dy = q2[1]-q1[1];
            T len = ceres::sqrt(dx*dx+dy*dy);
            T aw = T(1.0);

            if(len > T(1.0))
            {
                dx /= len;
                dy /= len;

                // angle weight
                T dotp = ceres::min(ceres::max(dx*observed_dir_l_x_+dy*observed_dir_l_y_,T(-0.9999)),T(0.9999));
                T angle = ceres::acos(dotp);

                if(!ceres::IsNaN(angle))
                {
                    if(angle > T(M_PI_2))
                        angle = T(M_PI)-angle;

                    aw = ceres::exp(2.0*angle);
                }
            }

            len = ceres::sqrt(len_sqr);
            l[0] = l[0]/len;
            l[1] = l[1]/len;
            l[2] = l[2]/len;

            T d = ceres::sqrt(l[0]*l[0]+l[1]*l[1]);

            residuals[0] = (l[0]*T(observed_pt1_x_)+l[1]*T(observed_pt1_y_)+l[2])/d*aw;
            residuals[1] = (l[0]*T(observed_pt2_x_)+l[1]*T(observed_pt2_y_)+l[2])/d*aw;

            return true;
        }

    private:
        double observed_pt1_x_;
        double observed_pt1_y_;
        double observed_pt2_x_;
        double observed_pt2_y_;
        double observed_dir_l_x_;
        double observed_dir_l_y_;
    };

    // length constraint for line bundling
    struct LineLengthConstraint
    {
        LineLengthConstraint(double observed_len):
            observed_len_(observed_len)
        {}

        template <typename T>
        bool operator()(const T* const camera,
                        const T* const line,
                        const T* const intrinsic,
                        T* residuals) const
        {
            // Translate into camera coordinate system
            T Q1[3];
            T Q2[3];

            Q1[0] = line[0]; Q1[1] = line[1]; Q1[2] = line[2];
            Q2[0] = line[3]; Q2[1] = line[4]; Q2[2] = line[5];

            T q1[3],q2[3];
            ceres::AngleAxisRotatePoint(camera, Q1, q1);
            ceres::AngleAxisRotatePoint(camera, Q2, q2);

            q1[0] += camera[3];
            q1[1] += camera[4];
            q1[2] += camera[5];

            q2[0] += camera[3];
            q2[1] += camera[4];
            q2[2] += camera[5];

            const T& px    = intrinsic[0];
            const T& py    = intrinsic[1];
            const T& fx    = intrinsic[2];
            const T& fy    = intrinsic[3];

            T x1n = (1.0 * q1[0] + 0.0 * q1[2]) / q1[2];
            T y1n = (1.0 * q1[1] + 0.0 * q1[2]) / q1[2];

            T x2n = (1.0 * q2[0] + 0.0 * q2[2]) / q2[2];
            T y2n = (1.0 * q2[1] + 0.0 * q2[2]) / q2[2];

            T xd1 = x1n;
            T yd1 = y1n;

            T xd2 = x2n;
            T yd2 = y2n;

            // projection function
            q1[0] = xd1;
            q1[1] = yd1;

            q2[0] = xd2;
            q2[1] = yd2;

            // projection function
            q1[0] = (fx * q1[0] + px);
            q1[1] = (fy * q1[1] + py);

            q2[0] = (fx * q2[0] + px);
            q2[1] = (fy * q2[1] + py);

            // length of backproj. segment
            T dx = q2[0]-q1[0];
            T dy = q2[1]-q1[1];
            T len = ceres::sqrt(dx*dx+dy*dy);

            residuals[0] = len-T(observed_len_);

            return true;
        }

    private:
        double observed_len_;
    };

    // optimizer using CERES
    class LineOptimizer
    {
    public:
        LineOptimizer(std::map<unsigned int,L3DPP::View*> views,
                      std::vector<L3DPP::LineCluster3D>* clusters3D,
                      const unsigned int max_iter) :
            views_(views), clusters3D_(clusters3D), max_iter_(max_iter){}

        // solve the bundling problem
        void optimize();

    private:
        std::map<unsigned int,L3DPP::View*> views_;
        std::vector<L3DPP::LineCluster3D>* clusters3D_;
        unsigned int max_iter_;
    };
}

#endif //L3DPP_CERES

#endif //I3D_LINE3D_PP_OPTIMIZATION_H_
