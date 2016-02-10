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
    const size_t LINE_SIZE = 4;
    const size_t INTRINSIC_SIZE = 4;
    const size_t CAM_PARAMETERS_SIZE = 6;
    const double LOSS_THRESHOLD = 2.0;

    // reprojection error for 3D line
    struct LineReprojectionError
    {
        LineReprojectionError(double observed_pt1_x, double observed_pt1_y,
                              double observed_pt2_x, double observed_pt2_y,
                              double observed_dir_x, double observed_dir_y):
            observed_pt1_x_(observed_pt1_x),
            observed_pt1_y_(observed_pt1_y),
            observed_pt2_x_(observed_pt2_x),
            observed_pt2_y_(observed_pt2_y),
            observed_norm_dir_x_(observed_dir_x),
            observed_norm_dir_y_(observed_dir_y)
        {}

        template <typename T>
        bool operator()(const T* const camera,
                        const T* const line,
                        const T* const intrinsic,
                        T* residuals) const
        {
            // convert to Plücker coordinates
            T sx = line[1]; T sy = line[2]; T sz = line[3];
            T omega = line[0];
            T nm = sx*sx+sy*sy+sz*sz;
            T div = T(1.0)/T(1.0+nm);

            T l[3];
            T m[3];

            l[0] = div * (T(1.0)-nm+T(2.0)*sx*sx);
            l[1] = div * (T(2.0)*sz+T(2.0)*sy*sx);
            l[2] = div * (T(-2.0)*sy+T(2.0)*sz*sx);

            m[0] = omega * div * (T(-2.0)*sz+T(2.0)*sx*sy);
            m[1] = omega * div * (T(1.0)-nm+T(2.0)*sy*sy);
            m[2] = omega * div * (T(2.0)*sx+T(2.0)*sz*sy);

            // check condition
            if(ceres::abs(omega) < 1e-12)
            {
                residuals[0] = T(0.0);
                residuals[1] = T(0.0);
                return false;
            }

            // Translate into camera coordinate system
            T Ccl[3]; // crossproduct: cam_center x l
            Ccl[0] = camera[4]*l[2] - camera[5]*l[1];
            Ccl[1] = -(camera[3]*l[2] - camera[5]*l[0]);
            Ccl[2] = camera[3]*l[1] - camera[4]*l[0];

            m[0] -= Ccl[0]; m[1] -= Ccl[1]; m[2] -= Ccl[2];

            T q[3];
            ceres::AngleAxisRotatePoint(camera, m, q);

            // project to image
            const T& px    = intrinsic[2];
            const T& py    = intrinsic[3];
            const T& fx    = intrinsic[0];
            const T& fy    = intrinsic[1];

            T proj_l[3];
            proj_l[0] = fy*q[0];
            proj_l[1] = fx*q[1];
            proj_l[2] = -fy*px*q[0]-fx*py*q[1]+fx*fy*q[2];

            // normalize line
            T len_sqr1 = proj_l[0]*proj_l[0]+proj_l[1]*proj_l[1];
            T d = ceres::sqrt(len_sqr1);

            if(d < 1e-12)
            {
                residuals[0] = T(0.0);
                residuals[1] = T(0.0);
                return false;
            }

            // angle constraint
            T dx = proj_l[0];
            T dy = proj_l[1];
            T aw = T(1.0);

            if(d > 1e-12)
            {
                dx /= d;
                dy /= d;

                // angle weight
                T dotp = ceres::min(ceres::max(dx*observed_norm_dir_x_+dy*observed_norm_dir_y_,T(-0.9999)),T(0.9999));
                T angle = ceres::acos(dotp);

                if(!ceres::IsNaN(angle))
                {
                    if(angle > T(M_PI_2))
                        angle = T(M_PI)-angle;

                    aw = ceres::exp(2.0*angle);
                }
            }

            residuals[0] = (proj_l[0]*T(observed_pt1_x_)+proj_l[1]*T(observed_pt1_y_)+proj_l[2])/d*aw;
            residuals[1] = (proj_l[0]*T(observed_pt2_x_)+proj_l[1]*T(observed_pt2_y_)+proj_l[2])/d*aw;

            return true;
        }

    private:
        double observed_pt1_x_;
        double observed_pt1_y_;
        double observed_pt2_x_;
        double observed_pt2_y_;
        double observed_norm_dir_x_;
        double observed_norm_dir_y_;
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
