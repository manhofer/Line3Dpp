#include "optimization.h"

#ifdef L3DPP_CERES

namespace L3DPP
{
    //------------------------------------------------------------------------------
    void LineOptimizer::optimize()
    {
        if(clusters3D_->size() == 0)
            return;

        // init CERES data structures
        size_t num_lines = clusters3D_->size();
        size_t num_cams = views_.size();

        double* lines = new double[num_lines * LINE_SIZE];
        double* tmp_pts = new double[num_lines * 6];
        double* cameras = new double[num_cams * CAM_PARAMETERS_SIZE];
        double* intrinsics = new double[num_cams * INTRINSIC_SIZE];

        // initialize problem
        ceres::Problem* problem = new ceres::Problem();

        std::vector<bool> keep_const(clusters3D_->size());

        // lines
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<clusters3D_->size(); ++i)
        {
            L3DPP::LineCluster3D LC = clusters3D_->at(i);

            // convert to Plücker
            Eigen::Vector3d l = LC.seg3D().P2()-LC.seg3D().P1();
            l.normalize();
            Eigen::Vector3d m = (0.5*(LC.seg3D().P1()+LC.seg3D().P2())).cross(l);

            // convert to Cayley [Zhang and Koch, J. Vis. Commun. Image R., 2014]
            Eigen::Matrix3d Q;
            Eigen::Vector3d e1,e2;
            if(m.norm() < L3D_EPS)
            {
                // compute nullspace of l'
                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(l.transpose());
                Eigen::MatrixXd e = lu_decomp.kernel();

                e1 = Eigen::Vector3d(e(0,0),e(1,0),e(2,0));
                e2 = Eigen::Vector3d(e(0,1),e(1,1),e(2,1));
            }
            else
            {
                e1 = m.normalized();
                e2 = (l.cross(m)).normalized();
            }

            Q(0,0) = l(0); Q(0,1) = e1(0); Q(0,2) = e2(0);
            Q(1,0) = l(1); Q(1,1) = e1(1); Q(1,2) = e2(1);
            Q(2,0) = l(2); Q(2,1) = e1(2); Q(2,2) = e2(2);

            Eigen::Matrix3d sx = (Q-Eigen::MatrixXd::Identity(3,3))*((Q+Eigen::MatrixXd::Identity(3,3)).inverse());

            Eigen::Vector3d s(sx(2,1),sx(0,2),sx(1,0));
            double omega = m.norm();

            if(std::isnan(s(0)) || std::isnan(s(1)) || std::isnan(s(2)) || std::isnan(omega))
            {
                // symmetric line coords... do not bundle
                lines[i * LINE_SIZE + 0] = -1;
                lines[i * LINE_SIZE + 1] = 0;
                lines[i * LINE_SIZE + 2] = 0;
                lines[i * LINE_SIZE + 3] = 0;

                // set constant
                keep_const[i] = true;
            }
            else
            {
                lines[i * LINE_SIZE + 0] = omega;
                lines[i * LINE_SIZE + 1] = s(0);
                lines[i * LINE_SIZE + 2] = s(1);
                lines[i * LINE_SIZE + 3] = s(2);

                // bundle
                keep_const[i] = false;
            }

            tmp_pts[i * 6 + 0] = LC.seg3D().P1().x();
            tmp_pts[i * 6 + 1] = LC.seg3D().P1().y();
            tmp_pts[i * 6 + 2] = LC.seg3D().P1().z();
            tmp_pts[i * 6 + 3] = LC.seg3D().P2().x();
            tmp_pts[i * 6 + 4] = LC.seg3D().P2().y();
            tmp_pts[i * 6 + 5] = LC.seg3D().P2().z();
        }

        // cameras & intrinsics
        std::map<unsigned int,size_t> cam_global2local;
        std::map<unsigned int,L3DPP::View*>::const_iterator it = views_.begin();
        for(size_t i=0; it!=views_.end(); ++it,++i)
        {
            // set local ID
            cam_global2local[it->first] = i;

            // camera (rotation and center)
            L3DPP::View* v = it->second;
            Eigen::Matrix3d rot = v->R();
            double rotation[9] =  {rot(0,0), rot(1,0), rot(2,0),
                                   rot(0,1), rot(1,1), rot(2,1),
                                   rot(0,2), rot(1,2), rot(2,2)};
            double axis_angle[3];
            ceres::RotationMatrixToAngleAxis(rotation, axis_angle);

            cameras[(i*CAM_PARAMETERS_SIZE) + 0] = axis_angle[0];
            cameras[(i*CAM_PARAMETERS_SIZE) + 1] = axis_angle[1];
            cameras[(i*CAM_PARAMETERS_SIZE) + 2] = axis_angle[2];

            cameras[(i*CAM_PARAMETERS_SIZE) + 3] = (v->C())[0];
            cameras[(i*CAM_PARAMETERS_SIZE) + 4] = (v->C())[1];
            cameras[(i*CAM_PARAMETERS_SIZE) + 5] = (v->C())[2];

            // intrinsics -> cof(K)
            double fx = (v->K())(0,0);
            double fy = (v->K())(1,1);
            double px = (v->K())(0,2);
            double py = (v->K())(1,2);

            intrinsics[(i*INTRINSIC_SIZE + 0)] = px;
            intrinsics[(i*INTRINSIC_SIZE + 1)] = py;
            intrinsics[(i*INTRINSIC_SIZE + 2)] = fx;
            intrinsics[(i*INTRINSIC_SIZE + 3)] = fy;
        }

        // store used camera pointers
        std::map<double*,bool> used_cams;
        std::map<double*,bool> used_intrinsics;

        // add residual blocks
        ceres::LossFunction* loss_function_lines = new ceres::HuberLoss(LOSS_THRESHOLD);
        ceres::ScaledLoss* scaled_loss_lines = new ceres::ScaledLoss(loss_function_lines,1.0,ceres::TAKE_OWNERSHIP);
        for(size_t i=0; i<clusters3D_->size(); ++i)
        {
            // iterate over 2D residuals
            std::list<L3DPP::Segment2D>::const_iterator it=clusters3D_->at(i).residuals()->begin();
            for(; it!=clusters3D_->at(i).residuals()->end(); ++it)
            {
                L3DPP::Segment2D seg2D = *it;
                size_t camera_idx = cam_global2local[seg2D.camID()];
                L3DPP::View* v = views_[seg2D.camID()];

                ceres::CostFunction* cost_function;

                // 2D line points and direction
                Eigen::Vector4f coords = v->getLineSegment2D(seg2D.segID());
                Eigen::Vector2d p1(coords.x(),coords.y());
                Eigen::Vector2d p2(coords.z(),coords.w());

                Eigen::Vector2d dir = (p2-p1).normalized();

                cost_function =  // 2 residuals, 6 camera parameters (ext), 4 line parameters
                    new ceres::AutoDiffCostFunction<LineReprojectionError, 2, CAM_PARAMETERS_SIZE, LINE_SIZE, INTRINSIC_SIZE>(
                            new LineReprojectionError(p1.x(),p1.y(),p2.x(),p2.y(),-dir.y(),dir.x())); // direction as normal vector!
                problem->AddResidualBlock(cost_function,scaled_loss_lines,
                                          cameras + camera_idx*CAM_PARAMETERS_SIZE,
                                          lines + i*LINE_SIZE, intrinsics + camera_idx*INTRINSIC_SIZE);

                used_cams[cameras + camera_idx*CAM_PARAMETERS_SIZE] = true;
                used_intrinsics[intrinsics + camera_idx*INTRINSIC_SIZE] = true;
            }
        }

        // set cameras and intrinsics as constant
        std::map<double*,bool>::const_iterator uc_it = used_cams.begin();
        for(; uc_it!=used_cams.end(); ++uc_it)
        {
            problem->SetParameterBlockConstant(uc_it->first);
        }
        uc_it = used_intrinsics.begin();
        for(; uc_it!=used_intrinsics.end(); ++uc_it)
        {
            problem->SetParameterBlockConstant(uc_it->first);
        }

        // set badly conditioned lines as constant
        unsigned int num_const = 0;
        for(size_t i=0; i<keep_const.size(); ++i)
        {
            if(keep_const[i])
            {
                problem->SetParameterBlockConstant(lines +i*LINE_SIZE);
                ++num_const;
            }
        }
        std::cout << prefix_ << "#unoptimizable_lines = " << num_const << std::endl;

        // solve
        ceres::Solver::Options options;
        options.max_num_iterations = max_iter_;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.num_threads = boost::thread::hardware_concurrency();
        options.minimizer_progress_to_stdout = true;
        options.num_linear_solver_threads = boost::thread::hardware_concurrency();

        ceres::Solver::Summary summary;
        ceres::Solve(options,problem,&summary);
        std::cout << summary.FullReport();

        // write back
        std::vector<L3DPP::LineCluster3D> clusters_copy = *clusters3D_;
        clusters3D_->clear();
        for(size_t i=0; i<clusters_copy.size(); ++i)
        {
            L3DPP::LineCluster3D LC = clusters_copy[i];

            // get final Cayley coords
            double omega = lines[i* LINE_SIZE + 0];
            Eigen::Vector3d s(lines[i * LINE_SIZE + 1],
                              lines[i * LINE_SIZE + 2],
                              lines[i * LINE_SIZE + 3]);

            // get old coords
            Eigen::Vector3d P1_old(tmp_pts[i * 6 + 0],
                                   tmp_pts[i * 6 + 1],
                                   tmp_pts[i * 6 + 2]);
            Eigen::Vector3d P2_old(tmp_pts[i * 6 + 3],
                                   tmp_pts[i * 6 + 4],
                                   tmp_pts[i * 6 + 5]);

            Eigen::Vector3d P1,P2;
            if(omega < 0.0 || fabs(omega) < L3D_EPS)
            {
                // keep original coords
                P1 = P1_old;
                P2 = P2_old;
            }
            else
            {
                // update coords
                Eigen::Matrix3d sx = Eigen::Matrix3d::Constant(0.0);
                sx(0,1) = -s.z(); sx(0,2) = s.y();
                sx(1,0) = s.z();  sx(1,2) = -s.x();
                sx(2,0) = -s.y(); sx(2,1) = s.x();

                double nm = s.x()*s.x()+s.y()*s.y()+s.z()*s.z();
                Eigen::Matrix3d Q = 1.0/(1.0+nm) * ((1.0-nm)*Eigen::Matrix3d::Identity() + 2.0*sx + 2.0*s*s.transpose());

                Eigen::Vector3d l(Q(0,0),Q(1,0),Q(2,0));
                Eigen::Vector3d m(Q(0,1),Q(1,1),Q(2,1));
                m *= omega;

                // convert back to P1,P2
                if(fabs(l.x()) > L3D_EPS || fabs(l.y()) > L3D_EPS || fabs(l.z()) > L3D_EPS)
                {
                    Eigen::Vector3d Pm = 0.5*(P1_old+P2_old);

                    double x1,x2,x3;
                    if(fabs(l.x()) > fabs(l.y()) && fabs(l.x()) > fabs(l.z()))
                    {
                        x1 = Pm.x();
                        x3 = (-m.y()-x1*l.z())/-l.x();
                        x2 = (m.z()-x1*l.y())/-l.x();
                    }
                    else if(fabs(l.y()) > fabs(l.x()) && fabs(l.y()) > fabs(l.z()))
                    {
                        x2 = Pm.y();
                        x3 = (m.x()-x2*l.z())/-l.y();
                        x1 = (m.z()+x2*l.x())/l.y();
                    }
                    else
                    {
                        x3 = Pm.z();
                        x2 = (m.x()+x3*l.y())/l.z();
                        x1 = (-m.y()+x3*l.x())/l.z();
                    }

                    Pm = Eigen::Vector3d(x1,x2,x3);
                    P1 = Pm+l;
                    P2 = Pm-l;
                }
                else
                {
                    // numerically unstable... keep unoptimized
                    P1 = P1_old;
                    P2 = P2_old;
                }
            }

            // check length
            if((P1-P2).norm() > L3D_EPS)
            {
                // still valid
                LC.update3Dline(L3DPP::Segment3D(P1,P2));
                clusters3D_->push_back(LC);
            }
        }

        // cleanup
        delete lines;
        delete tmp_pts;
        delete cameras;
        delete intrinsics;
        delete problem;
    }
}

#endif //L3DPP_CERES
