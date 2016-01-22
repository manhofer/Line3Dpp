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
        double* cameras = new double[num_cams * CAM_PARAMETERS_SIZE];
        double* intrinsics = new double[num_cams * INTRINSIC_SIZE];

        // initialize problem
        ceres::Problem* problem = new ceres::Problem();

        // lines
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<clusters3D_->size(); ++i)
        {
            L3DPP::LineCluster3D LC = clusters3D_->at(i);
            lines[i * LINE_SIZE + 0] = LC.seg3D().P1().x();
            lines[i * LINE_SIZE + 1] = LC.seg3D().P1().y();
            lines[i * LINE_SIZE + 2] = LC.seg3D().P1().z();
            lines[i * LINE_SIZE + 3] = LC.seg3D().P2().x();
            lines[i * LINE_SIZE + 4] = LC.seg3D().P2().y();
            lines[i * LINE_SIZE + 5] = LC.seg3D().P2().z();
        }

        // cameras & intrinsics
        std::map<unsigned int,size_t> cam_global2local;
        std::map<unsigned int,L3DPP::View*>::iterator it = views_.begin();
        for(size_t i=0; it!=views_.end(); ++it,++i)
        {
            // set local ID
            cam_global2local[it->first] = i;

            // camera
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

            cameras[(i*CAM_PARAMETERS_SIZE) + 3] = (v->t())[0];
            cameras[(i*CAM_PARAMETERS_SIZE) + 4] = (v->t())[1];
            cameras[(i*CAM_PARAMETERS_SIZE) + 5] = (v->t())[2];

            // intrinsics
            intrinsics[(i*INTRINSIC_SIZE + 0)] = (v->K())(0,2); //px
            intrinsics[(i*INTRINSIC_SIZE + 1)] = (v->K())(1,2); //py
            intrinsics[(i*INTRINSIC_SIZE + 2)] = (v->K())(0,0); //fx
            intrinsics[(i*INTRINSIC_SIZE + 3)] = (v->K())(1,1); //fy
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
            std::list<L3DPP::Segment2D>::iterator it=clusters3D_->at(i).residuals()->begin();
            for(; it!=clusters3D_->at(i).residuals()->end(); ++it)
            {
                L3DPP::Segment2D seg2D = *it;
                size_t camera_idx = cam_global2local[seg2D.camID()];
                L3DPP::View* v = views_[seg2D.camID()];

                ceres::CostFunction* cost_function;

                // line equation
                Eigen::Vector4f coords = v->getLineSegment2D(seg2D.segID());
                Eigen::Vector2d p1(coords.x(),coords.y());
                Eigen::Vector2d p2(coords.z(),coords.w());

                Eigen::Vector2d dir = p2-p1;
                dir.normalize();

                cost_function =  // 2 residuals, 6 camera parameter (ext), 6 line parameter
                    new ceres::AutoDiffCostFunction<LineReprojectionError, 2, CAM_PARAMETERS_SIZE, LINE_SIZE, INTRINSIC_SIZE>(
                            new LineReprojectionError(p1.x(),p1.y(),p2.x(),p2.y(),dir.x(),dir.y()));
                problem->AddResidualBlock(cost_function,scaled_loss_lines,
                                          cameras + camera_idx*CAM_PARAMETERS_SIZE,
                                          lines + i*LINE_SIZE, intrinsics + camera_idx*INTRINSIC_SIZE);

                used_cams[cameras + camera_idx*CAM_PARAMETERS_SIZE] = true;
                used_intrinsics[intrinsics + camera_idx*INTRINSIC_SIZE] = true;
            }

            // add length constraint
            L3DPP::Segment2D corr_seg = clusters3D_->at(i).correspondingSeg2D();
            L3DPP::View* v = views_[corr_seg.camID()];
            Eigen::Vector4f coords = v->getLineSegment2D(corr_seg.segID());
            size_t camera_idx = cam_global2local[corr_seg.camID()];
            double length = (Eigen::Vector2d(coords(0),coords(1))-Eigen::Vector2d(coords(2),coords(3))).norm();

            ceres::LossFunction* loss_function_line_endpoints = new ceres::HuberLoss(LOSS_THRESHOLD);
            ceres::ScaledLoss* scaled_loss_line_endpoints = new ceres::ScaledLoss(loss_function_line_endpoints,
                                                                                  1.0,ceres::TAKE_OWNERSHIP);

            ceres::CostFunction* cost_function;

            cost_function =  // 1 residual
                new ceres::AutoDiffCostFunction<LineLengthConstraint, 1, CAM_PARAMETERS_SIZE, LINE_SIZE, INTRINSIC_SIZE>(
                        new LineLengthConstraint(length));
            problem->AddResidualBlock(cost_function,scaled_loss_line_endpoints,
                                      cameras + camera_idx*CAM_PARAMETERS_SIZE,
                                      lines + i*LINE_SIZE, intrinsics + camera_idx*INTRINSIC_SIZE);

            used_cams[cameras + camera_idx*CAM_PARAMETERS_SIZE] = true;
            used_intrinsics[intrinsics + camera_idx*INTRINSIC_SIZE] = true;
        }

        // set cameras and intrinsics as constant
        std::map<double*,bool>::iterator uc_it = used_cams.begin();
        for(; uc_it!=used_cams.end(); ++uc_it)
        {
            problem->SetParameterBlockConstant(uc_it->first);
        }
        uc_it = used_intrinsics.begin();
        for(; uc_it!=used_intrinsics.end(); ++uc_it)
        {
            problem->SetParameterBlockConstant(uc_it->first);
        }

        // solve
        ceres::Solver::Options options;
        options.max_num_iterations = max_iter_;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.num_threads = boost::thread::hardware_concurrency();
        options.minimizer_progress_to_stdout = true;
        options.num_linear_solver_threads =  boost::thread::hardware_concurrency();

        ceres::Solver::Summary summary;
        ceres::Solve(options,problem,&summary);
        std::cout << summary.FullReport();

        // write back
        std::vector<L3DPP::LineCluster3D> clusters_copy = *clusters3D_;
        clusters3D_->clear();
        for(size_t i=0; i<clusters_copy.size(); ++i)
        {
            L3DPP::LineCluster3D LC = clusters_copy[i];

            // check bundled line
            Eigen::Vector3d P1(lines[i * LINE_SIZE + 0],
                               lines[i * LINE_SIZE + 1],
                               lines[i * LINE_SIZE + 2]);
            Eigen::Vector3d P2(lines[i * LINE_SIZE + 3],
                               lines[i * LINE_SIZE + 4],
                               lines[i * LINE_SIZE + 5]);

            if((P1-P2).norm() > L3D_EPS)
            {
                // still valid
                LC.update3Dline(L3DPP::Segment3D(P1,P2));
                clusters3D_->push_back(LC);
            }
        }

        // cleanup
        delete lines;
        delete cameras;
        delete intrinsics;
        delete problem;
    }
}

#endif //L3DPP_CERES
