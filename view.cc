#include "view.h"

namespace L3DPP
{
    //------------------------------------------------------------------------------
    View::View(const unsigned int id, L3DPP::DataArray<float4>* lines,
               const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
               const Eigen::Vector3d& t,
               const unsigned int width, const unsigned int height,
               const float median_depth,
               L3DPP::DataArray<float>* superpixels) :
        id_(id), lines_(lines), K_(K), R_(R), t_(t),
        width_(width), height_(height), initial_median_depth_(fmax(fabs(median_depth),L3D_EPS)),
        superpixels_(superpixels)
    {
        // init
        diagonal_ = sqrtf(float(width_*width_+height_*height_));
        min_line_length_ = diagonal_*L3D_DEF_MIN_LINE_LENGTH_FACTOR;

        collin_t_ = 0.0f;

        // camera
        pp_ = Eigen::Vector3d(K_(0,2),K_(1,2),1.0);

        Kinv_ = K_.inverse();
        Rt_ = R_.transpose();
        RtKinv_  = Rt_*Kinv_;
        C_ = Rt_ * (-1.0 * t_);

        k_ = 0.0f;
        median_depth_ = 0.0f;
        median_sigma_ = 0.0f;

#ifdef L3DPP_CUDA
        C_f3_ = make_float3(C_.x(),C_.y(),C_.z());
        // RtKinv -> data array
        RtKinv_DA_ = new L3DPP::DataArray<float>(3,3);
        for(size_t r=0; r<3; ++r)
            for(size_t c=0; c<3; ++c)
                RtKinv_DA_->dataCPU(c,r)[0] = RtKinv_(r,c);
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    View::~View()
    {
        if(lines_ != NULL)
            delete lines_;

        if(superpixels_ != NULL)
            delete superpixels_;

#ifdef L3DPP_CUDA
        if(RtKinv_DA_ != NULL)
            delete RtKinv_DA_;
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void View::drawLineImage(cv::Mat& img)
    {
        img = cv::Mat::zeros(height_,width_,CV_8UC3);

        for(size_t i=0; i<lines_->width(); ++i)
        {
            float4 coords = lines_->dataCPU(i,0)[0];

            cv::Point p1(coords.x,coords.y);
            cv::Point p2(coords.z,coords.w);
            cv::line(img,p1,p2,cv::Scalar(255,255,255),3);
        }
    }

    //------------------------------------------------------------------------------
    void View::drawSingleLine(const unsigned int id, cv::Mat& img,
                              const cv::Scalar& color)
    {
        if(id < lines_->width())
        {
            float4 coords = lines_->dataCPU(id,0)[0];
            cv::Point p1(coords.x,coords.y);
            cv::Point p2(coords.z,coords.w);
            cv::line(img,p1,p2,color,3);
        }
    }

    //------------------------------------------------------------------------------
    void View::drawEpipolarLine(const Eigen::Vector3d& epi, cv::Mat& img)
    {
        // intersect with image borders
        Eigen::Vector3d p1(0,0,1);
        Eigen::Vector3d p2(img.cols,0,1);
        Eigen::Vector3d p3(img.cols,img.rows,1);
        Eigen::Vector3d p4(0,img.rows,1);

        Eigen::Vector3d borders[4];
        borders[0] = p1.cross(p2);
        borders[1] = p2.cross(p3);
        borders[2] = p3.cross(p4);
        borders[3] = p4.cross(p1);

        std::vector<Eigen::Vector3d> intersections;
        for(size_t i=0; i<4; ++i)
        {
            Eigen::Vector3d I = borders[i].cross(epi);
            if(fabs(I.z()) > L3D_EPS)
            {
                I /= I.z();
                I(2) = 1.0;

                // check position
                if(I.x() > -1.0 && I.x() < img.cols+1 &&
                        I.y() > -1.0 && I.y() < img.rows+1)
                {
                    intersections.push_back(I);
                }
            }
        }

        if(intersections.size() < 2)
            return;

        // find intersections that are farthest apart
        double max_dist = 0.0f;
        Eigen::Vector3d e_p1(0,0,0);
        Eigen::Vector3d e_p2(0,0,0);

        for(size_t i=0; i<intersections.size()-1; ++i)
        {
            Eigen::Vector3d _p = intersections[i];
            for(size_t j = i+1; j<intersections.size(); ++j)
            {
                Eigen::Vector3d _q = intersections[j];
                double len = (_p-_q).norm();
                if(len > max_dist)
                {
                    max_dist = len;
                    e_p1 = _p;
                    e_p2 = _q;
                }
            }
        }

        cv::Point pt1(e_p1.x(),e_p1.y());
        cv::Point pt2(e_p2.x(),e_p2.y());
        cv::line(img,pt1,pt2,cv::Scalar(0,255,255),3);
    }

    //------------------------------------------------------------------------------
    void View::findCollinearSegments(const float dist_t, bool useGPU)
    {
        if(fabs(dist_t-collin_t_) < L3D_EPS)
        {
            // already computed
            return;
        }

        if(dist_t > L3D_EPS)
        {
#ifndef L3DPP_CUDA
            useGPU = false;
#endif //L3DPP_CUDA
            collin_t_ = dist_t;

            if(useGPU)
                findCollinGPU();
            else
                findCollinCPU();
        }
    }

    //------------------------------------------------------------------------------
    void View::findCollinGPU()
    {
        // reset
        collin_ = std::vector<std::list<unsigned int> >(lines_->width());

#ifdef L3DPP_CUDA
        // upload
        lines_->upload();

        // buffer
        L3DPP::DataArray<char>* buffer = new L3DPP::DataArray<char>(lines_->width(),
                                                                    lines_->width(),true);

        // GPU function
        L3DPP::find_collinear_segments_GPU(buffer,lines_,collin_t_);
        buffer->download();
        buffer->removeFromGPU();

        // process
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<collin_.size(); ++i)
        {
            for(size_t c=0; c<buffer->width(); ++c)
            {
                char data = buffer->dataCPU(c,i)[0];
                if(data == 1)
                    collin_[i].push_back(c);
            }
        }

        // cleanup
        delete buffer;
        lines_->removeFromGPU();
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void View::findCollinCPU()
    {
        // reset
        collin_ = std::vector<std::list<unsigned int> >(lines_->width());

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int r=0; r<collin_.size(); ++r)
        {
            Eigen::Vector3d p[2];
            float4 l1 = lines_->dataCPU(r,0)[0];

            p[0] = Eigen::Vector3d(l1.x,l1.y,1.0f);
            p[1] = Eigen::Vector3d(l1.z,l1.w,1.0f);
            Eigen::Vector3d line1 = p[0].cross(p[1]);

            for(size_t c=0; c<lines_->width(); ++c)
            {
                if(r == c)
                    continue;

                // line data
                float4 l2 = lines_->dataCPU(c,0)[0];

                Eigen::Vector3d q[2];
                q[0] = Eigen::Vector3d(l2.x,l2.y,1.0f);
                q[1] = Eigen::Vector3d(l2.z,l2.w,1.0f);
                Eigen::Vector3d line2 = q[0].cross(q[1]);

                // check location (overlap)
                if(pointOnSegment(p[0],p[1],q[0]) ||
                        pointOnSegment(p[0],p[1],q[1]) ||
                        pointOnSegment(q[0],q[1],p[0]) ||
                        pointOnSegment(q[0],q[1],p[1]))
                {
                    // overlap -> not collinear
                    continue;
                }

                // compute distances
                float d1 = fmax(distance_point2line_2D(line1,q[0]),
                                distance_point2line_2D(line1,q[1]));
                float d2 = fmax(distance_point2line_2D(line2,p[0]),
                                distance_point2line_2D(line2,p[1]));

                if(fmax(d1,d2) < collin_t_)
                {
                    collin_[r].push_back(c);
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    float View::distance_point2line_2D(const Eigen::Vector3d& line, const Eigen::Vector3d& p)
    {
        return fabs((line.x()*p.x()+line.y()*p.y()+line.z())/sqrtf(line.x()*line.x()+line.y()*line.y()));
    }

    //------------------------------------------------------------------------------
    float View::smallerAngle(const Eigen::Vector2d& v1, const Eigen::Vector2d& v2)
    {
        float angle = acos(fmax(fmin(v1.dot(v2),1.0f),-1.0f));
        if(angle > L3D_PI_1_2)
            angle = M_PI-angle;

        return angle;
    }

    //------------------------------------------------------------------------------
    std::list<unsigned int> View::collinearSegments(const unsigned int segID)
    {
        if(collin_.size() == lines_->width() && segID < lines_->width())
            return collin_[segID];
        else
            return std::list<unsigned int>();
    }

    //------------------------------------------------------------------------------
    bool View::pointOnSegment(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                              const Eigen::Vector3d& x)
    {
        Eigen::Vector2d v1(p1.x()-x.x(),p1.y()-x.y());
        Eigen::Vector2d v2(p2.x()-x.x(),p2.y()-x.y());
        return (v1.dot(v2) < L3D_EPS);
    }

    //------------------------------------------------------------------------------
    void View::computeSpatialRegularizer(const float r)
    {
        k_ = getSpecificSpatialReg(r);
    }

    //------------------------------------------------------------------------------
    float View::getSpecificSpatialReg(const float r)
    {
        Eigen::Vector3d pp_shifted = pp_+Eigen::Vector3d(r,0.0,0.0);
        Eigen::Vector3d ray_pp = getNormalizedRay(pp_);
        Eigen::Vector3d ray_pp_shifted = getNormalizedRay(pp_shifted);
        double alpha = acos(fmin(fmax(double(ray_pp.dot(ray_pp_shifted)),-1.0),1.0));
        return sin(alpha);
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedRay(const Eigen::Vector3d& p)
    {
        Eigen::Vector3d ray = RtKinv_*p;
        return ray.normalized();
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedRay(const Eigen::Vector2d& p)
    {
        return getNormalizedRay(Eigen::Vector3d(p.x(),p.y(),1.0));
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedLinePointRay(const unsigned int lID,
                                                    const bool pt1)
    {
        Eigen::Vector3d ray(0,0,0);
        if(lID < lines_->width())
        {
            Eigen::Vector3d p;
            if(pt1)
            {
                // ray through P1
                p = Eigen::Vector3d(lines_->dataCPU(lID,0)[0].x,
                                    lines_->dataCPU(lID,0)[0].y,1.0);
            }
            else
            {
                // ray through P2
                p = Eigen::Vector3d(lines_->dataCPU(lID,0)[0].z,
                                    lines_->dataCPU(lID,0)[0].w,1.0);
            }

            return getNormalizedRay(p);
        }
        return ray;
    }

    //------------------------------------------------------------------------------
    L3DPP::Segment3D View::unprojectSegment(const unsigned int segID, const float depth1,
                                            const float depth2)
    {
        L3DPP::Segment3D seg3D;
        if(segID < lines_->width())
        {
            Eigen::Vector3d p1(lines_->dataCPU(segID,0)[0].x,
                               lines_->dataCPU(segID,0)[0].y,1.0);
            Eigen::Vector3d p2(lines_->dataCPU(segID,0)[0].z,
                               lines_->dataCPU(segID,0)[0].w,1.0);

            seg3D = L3DPP::Segment3D(C_ + getNormalizedRay(p1)*depth1,
                                     C_ + getNormalizedRay(p2)*depth2);
        }
        return seg3D;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector2d View::project(const Eigen::Vector3d& P)
    {
        Eigen::Vector3d q = (R_*P + t_);

        // projection to unit focal plane
        double xn = (1.0 * q[0] + 0.0 * q[2]) / q[2];
        double yn = (1.0 * q[1] + 0.0 * q[2]) / q[2];

        // projection function
        q[0] = xn;
        q[1] = yn;
        q[2] = 1;
        q = K_*q;

        Eigen::Vector2d res;
        res(0) = q(0)/q(2);
        res(1) = q(1)/q(2);
        return res;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::projectWithCheck(const Eigen::Vector3d& P)
    {
        Eigen::Vector3d q = (R_*P + t_);

        // projection to unit focal plane
        double xn = (1.0 * q[0] + 0.0 * q[2]) / q[2];
        double yn = (1.0 * q[1] + 0.0 * q[2]) / q[2];

        // projection function
        q[0] = xn;
        q[1] = yn;
        q[2] = 1;
        q = K_*q;

        Eigen::Vector3d res(0,0,-1);

        if(fabs(q(2)) > L3D_EPS)
        {
            res(0) = q(0)/q(2);
            res(1) = q(1)/q(2);
            res(2) = 1;
        }

        return res;
    }

    //------------------------------------------------------------------------------
    bool View::projectedLongEnough(const L3DPP::Segment3D& seg3D)
    {
        Eigen::Vector2d p1 = project(seg3D.P1());
        Eigen::Vector2d p2 = project(seg3D.P2());
        return ((p1-p2).norm() > min_line_length_);
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f View::getLineSegment2D(const unsigned int id)
    {
        Eigen::Vector4f coords(0,0,0,0);
        if(id < lines_->width())
        {
            float4 c = lines_->dataCPU(id,0)[0];
            coords(0) = c.x;
            coords(1) = c.y;
            coords(2) = c.z;
            coords(3) = c.w;
        }
        return coords;
    }

    //------------------------------------------------------------------------------
    float View::regularizerFrom3Dpoint(const Eigen::Vector3d& P)
    {
        return (P-C_).norm()*k_;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getOpticalAxis()
    {
        return getNormalizedRay(pp_);
    }

    //------------------------------------------------------------------------------
    double View::opticalAxesAngle(L3DPP::View* v)
    {
        Eigen::Vector3d r1 = getNormalizedRay(pp_);
        Eigen::Vector3d r2 = v->getOpticalAxis();

        return acos(fmin(fmax(double(r1.dot(r2)),-1.0),1.0));
    }

    //------------------------------------------------------------------------------
    double View::segmentQualityAngle(const L3DPP::Segment3D& seg3D,
                                     const unsigned int segID)
    {
        if(segID < lines_->width())
        {
            Eigen::Vector2d p1(lines_->dataCPU(segID,0)[0].x,
                               lines_->dataCPU(segID,0)[0].y);
            Eigen::Vector2d p2(lines_->dataCPU(segID,0)[0].z,
                               lines_->dataCPU(segID,0)[0].w);
            Eigen::Vector2d p = 0.5*(p1+p2);

            Eigen::Vector3d r1 = getNormalizedRay(p);
            Eigen::Vector3d r2 = seg3D.dir();

            return acos(fmin(fmax(double(r1.dot(r2)),-1.0),1.0));
        }

        return 0.0;
    }

    //------------------------------------------------------------------------------
    float View::distanceVisualNeighborScore(L3DPP::View* v)
    {
        // bring tgt camera center to src coordinate frame
        Eigen::Vector3d Ctgt_t = R_*v->C()+t_;

        // define two planes trough the camera center
        Eigen::Vector3d n1(1,0,0);
        Eigen::Vector3d n2(0,1,0);

        // compute distances to the planes
        float dist1 = fabs(n1.dot(Ctgt_t));
        float dist2 = fabs(n2.dot(Ctgt_t));

        return dist1+dist2;
    }

    //------------------------------------------------------------------------------
    float View::baseLine(L3DPP::View* v)
    {
        return (C_ - v->C()).norm();
    }

    //------------------------------------------------------------------------------
    void View::translate(const Eigen::Vector3d& t)
    {
        C_ += t;
        t_ = -R_ * C_;
    }
}
