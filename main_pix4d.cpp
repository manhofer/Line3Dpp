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

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"

// helper function for point triangulation
Eigen::Vector3d linearHomTriangulation(std::list<std::pair<size_t,Eigen::Vector2d> >& obs,
                                       std::vector<Eigen::MatrixXd>& P)
{
    if(obs.size() == 0 || P.size() == 0)
        return Eigen::Vector3d(0,0,0);

    std::vector<Eigen::MatrixXd> Sx(obs.size(), Eigen::MatrixXd::Zero(2,3));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(obs.size() * 2 , 4);
    std::list<std::pair<size_t,Eigen::Vector2d> >::iterator it = obs.begin();
    for(size_t i=0; it!=obs.end(); ++i,++it)
    {
        Eigen::Vector2d pt = (*it).second;
        size_t camID = (*it).first;

        Sx[i](0,1) = -1; Sx[i](0,2) = pt.y();
        Sx[i](1,0) =  1; Sx[i](1,2) = -pt.x();

        A.block<2,4>(i*2,0) = Sx[i] * P[camID];
    }

    Eigen::MatrixXd AtA(4, 4);
    AtA = A.transpose() * A;

    Eigen::MatrixXd U,V;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(AtA, Eigen::ComputeThinU | Eigen::ComputeThinV);

    U = svd.matrixU();
    V = svd.matrixV();

    Eigen::VectorXd X;

    X = V.col(3);
    X /= X(3);

    return Eigen::Vector3d(X(0),X(1),X(2));
}

// INFO:
// This executable reads Pix4D results (<project_prefix>/1_initial/params/*.txt) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the result file, you need to use the _original_ (distorted) images!
//
// NOTE:
// The algorithm takes the camera poses from the <project_prefix>_calibrated_camera_parameters.txt file,
// which means they are metrically correct, but in a local coordinate system! If you want to view the 3D lines
// together with the georeferenced 3D points from Pix4D you need to apply the appropriate transformation.

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, ".", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> pix4dArg("b", "params_folder", "folder containing the proeject files <project_prefix>_calibrated_camera_parameters.txt and <project_prefix>_tp_pix4d.txt", true, "", "string");
    cmd.add(pix4dArg);

    TCLAP::ValueArg<std::string> prefixArg("f", "project_prefix", "project name and output file prefix", true, "", "string");
    cmd.add(prefixArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> image_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);

    TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
    cmd.add(sigma_A_Arg);

    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
    cmd.add(sigma_P_Arg);

    TCLAP::ValueArg<float> epipolarArg("e", "min_epipolar_overlap", "minimum epipolar overlap for matching", false, L3D_DEF_EPIPOLAR_OVERLAP, "float");
    cmd.add(epipolarArg);

    TCLAP::ValueArg<int> knnArg("k", "knn_matches", "number of matches to be kept (<= 0 --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
    cmd.add(knnArg);

    TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
    cmd.add(segNumArg);

    TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
    cmd.add(visibilityArg);

    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);

    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);

    TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
    cmd.add(collinArg);

    TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
    cmd.add(cudaArg);

    TCLAP::ValueArg<bool> ceresArg("c", "use_ceres", "use CERES (for 3D line optimization)", false, L3D_DEF_USE_CERES, "bool");
    cmd.add(ceresArg);

    TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
    cmd.add(constRegDepthArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string imageFolder = inputArg.getValue().c_str();
    std::string paramsFolder = pix4dArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    std::string projextPrefix = prefixArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = imageFolder+"/Line3D++/";

    int maxWidth = scaleArg.getValue();
    unsigned int neighbors = std::max(neighborArg.getValue(),2);
    bool diffusion = diffusionArg.getValue();
    bool loadAndStore = loadArg.getValue();
    float collinearity = collinArg.getValue();
    bool useGPU = cudaArg.getValue();
    bool useCERES = ceresArg.getValue();
    float epipolarOverlap = fmin(fabs(epipolarArg.getValue()),0.99f);
    float sigmaA = fabs(sigma_A_Arg.getValue());
    float sigmaP = sigma_P_Arg.getValue();
    int kNN = knnArg.getValue();
    unsigned int maxNumSegments = segNumArg.getValue();
    unsigned int visibility_t = visibilityArg.getValue();
    float constRegDepth = constRegDepthArg.getValue();

    // check if parameter files exist
    std::string params_prefix = paramsFolder+"/"+projextPrefix;
    if(params_prefix.substr(params_prefix.length()-1,1) != "_")
        params_prefix += "_";

    std::string file1 = params_prefix+"calibrated_camera_parameters.txt";
    std::string file2 = params_prefix+"tp_pix4d.txt";
    boost::filesystem::path pf1(file1);
    boost::filesystem::path pf2(file2);
    if(!boost::filesystem::exists(pf1) || !boost::filesystem::exists(pf2))
    {
        std::cerr << "pix4d file '" << file1 << "' or '" << std::endl << file2 << "' does not exist!" << std::endl;
        return -1;
    }

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU);

    // camera parameter file
    std::ifstream pix4d_cam_file;
    pix4d_cam_file.open(file1.c_str());

    std::string pix4d_cam_line;
    // ignore descriptions...
    while(std::getline(pix4d_cam_file,pix4d_cam_line))
    {
        if(pix4d_cam_line.length() < 2)
            break;
    }

    // read camera data (sequentially)
    std::map<std::string,size_t> img2pos;
    std::map<size_t,std::string> pos2img;
    std::vector<std::string> cams_filenames;
    std::vector<Eigen::Matrix3d> cams_rotation;
    std::vector<Eigen::Matrix3d> cams_intrinsic;
    std::vector<Eigen::MatrixXd> cams_projection;
    std::vector<Eigen::Vector3d> cams_translation;
    std::vector<Eigen::Vector3d> cams_radial_dist;
    std::vector<Eigen::Vector2d> cams_tangential_dist;
    while(std::getline(pix4d_cam_file,pix4d_cam_line))
    {
        if(pix4d_cam_line.length() < 5)
            break;

        // filename
        std::stringstream pix4d_stream(pix4d_cam_line);
        std::string filename,width,height;
        pix4d_stream >> filename >> width >> height;

        size_t lastindex = filename.find_last_of(".");
        std::string rawname = filename.substr(0, lastindex);

        img2pos[rawname] = cams_filenames.size();
        pos2img[cams_filenames.size()] = rawname;
        cams_filenames.push_back(filename);

        // intrinsics
        Eigen::Matrix3d K;
        for(size_t i=0; i<3; ++i)
        {
            std::getline(pix4d_cam_file,pix4d_cam_line);
            pix4d_stream.clear();
            pix4d_stream.str(pix4d_cam_line);
            pix4d_stream >> K(i,0) >> K(i,1) >> K(i,2);
        }
        cams_intrinsic.push_back(K);

        // radial distortion
        Eigen::Vector3d radial;
        std::getline(pix4d_cam_file,pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> radial(0) >> radial(1) >> radial(2);
        cams_radial_dist.push_back(radial);

        // tangential distortion
        Eigen::Vector2d tangential;
        std::getline(pix4d_cam_file,pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> tangential(0) >> tangential(1);
        cams_tangential_dist.push_back(tangential);

        // translation
        Eigen::Vector3d t;
        std::getline(pix4d_cam_file,pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> t(0) >> t(1) >> t(2);

        // rotation
        Eigen::Matrix3d R;
        for(size_t i=0; i<3; ++i)
        {
            std::getline(pix4d_cam_file,pix4d_cam_line);
            pix4d_stream.clear();
            pix4d_stream.str(pix4d_cam_line);
            pix4d_stream >> R(i,0) >> R(i,1) >> R(i,2);
        }
        cams_rotation.push_back(R);

        t = -R*t;
        cams_translation.push_back(t);

        // projection
        Eigen::MatrixXd P(3,4);
        P.block<3,3>(0,0) = R;
        P.block<3,1>(0,3) = t;
        P = K*P;
        cams_projection.push_back(P);
    }
    pix4d_cam_file.close();

    // camera parameter file
    std::ifstream pix4d_point_file;
    pix4d_point_file.open(file2.c_str());

    std::string pix4d_point_line;

    // read point data
    std::map<std::string,std::list<unsigned int> > featuresPerCam;
    std::map<std::string,unsigned int> feat_key2id;
    std::map<unsigned int,std::string> feat_id2key;
    std::map<unsigned int,bool> feat_valid;
    std::map<unsigned int,Eigen::Vector3d> feat_pos3D;
    std::vector<std::list<std::pair<size_t,Eigen::Vector2d> > > feat_observations;

    std::string key;
    size_t key_img_pos;
    while(std::getline(pix4d_point_file,pix4d_point_line))
    {
        std::string id,rest;
        double px,py,scale;

        std::stringstream pix4d_stream(pix4d_point_line);
        pix4d_stream >> id >> rest;

        if(id.length() < 2)
            break;

        if(id.substr(0,1) != "-")
        {
            if(rest.length() == 0)
            {
                // new key image
                key = id;
                key_img_pos = img2pos[key];
            }
            else
            {
                // new feature for current key image
                pix4d_stream.clear();
                pix4d_stream.str(pix4d_point_line);
                pix4d_stream >> id >> px >> py >> scale;

                // check for new feature
                size_t fID;
                if(feat_key2id.find(id) == feat_key2id.end())
                {
                    // new feature
                    fID = feat_observations.size();
                    feat_key2id[id] = fID;
                    feat_id2key[fID] = id;
                    feat_valid[fID] = false;
                    feat_pos3D[fID] = Eigen::Vector3d(0,0,0);

                    feat_observations.push_back(std::list<std::pair<size_t,Eigen::Vector2d> >());
                }
                else
                {
                    // existing feature
                    fID = feat_key2id[id];
                }

                // add observation
                featuresPerCam[key].push_back(fID);
                feat_observations[fID].push_back(std::pair<size_t,Eigen::Vector2d>(key_img_pos,
                                                                                   Eigen::Vector2d(px,py)));
            }
        }
    }
    pix4d_point_file.close();

    std::cout << "Pix4D: #cameras = " << img2pos.size() << std::endl;
    std::cout << "Pix4D: #points  = " << feat_observations.size() << std::endl;

    // triangulate points (parallel)
    std::cout << "triangulating..." << std::endl;
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<feat_observations.size(); ++i)
    {
        std::list<std::pair<size_t,Eigen::Vector2d> > obs = feat_observations[i];
        if(obs.size() > 2)
        {
            Eigen::Vector3d P = linearHomTriangulation(obs,cams_projection);

            if(P.norm() > L3D_EPS)
            {
                feat_valid[i] = true;
                feat_pos3D[i] = P;
            }
        }
    }

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<cams_rotation.size(); ++i)
    {
        // load image
        std::string img_filename = imageFolder+"/"+cams_filenames[i];
        cv::Mat image = cv::imread(img_filename,CV_LOAD_IMAGE_GRAYSCALE);

        std::string key = pos2img[i];

        if(featuresPerCam.find(key) != featuresPerCam.end())
        {
            // camera center
            Eigen::Matrix3d Rt = cams_rotation[i].transpose();
            Eigen::Vector3d C = Rt * (-1.0 * cams_translation[i]);

            // compute median depth
            std::vector<float> depths;
            std::list<unsigned int> wpIDs = featuresPerCam[key];
            std::list<unsigned int>::iterator it = wpIDs.begin();
            for(; it!=wpIDs.end(); ++it)
            {
                if(feat_valid[*it])
                {
                    Eigen::Vector3d P = feat_pos3D[*it];
                    depths.push_back((P-C).norm());
                }
            }

            if(depths.size() > 2)
            {
                std::sort(depths.begin(),depths.end());
                float med_depth = depths[depths.size()/2];

                // undistort
                cv::Mat img_undist;
                Line3D->undistortImage(image,img_undist,cams_radial_dist[i],
                                       cams_tangential_dist[i],cams_intrinsic[i]);

                // add to system
                Line3D->addImage(i,img_undist,cams_intrinsic[i],cams_rotation[i],
                                 cams_translation[i],med_depth,wpIDs);
            }
        }
    }

    // match images
    Line3D->matchImages(sigmaP,sigmaA,neighbors,epipolarOverlap,
                        kNN,constRegDepth);

    // compute result
    Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useCERES);

    // save end result
    std::vector<L3DPP::FinalLine3D> result;
    Line3D->get3Dlines(result);

    // save as STL
    Line3D->saveResultAsSTL(outputFolder);
    // save as OBJ
    Line3D->saveResultAsOBJ(outputFolder);
    // save as TXT
    Line3D->save3DLinesAsTXT(outputFolder);
    // save as BIN
    Line3D->save3DLinesAsBIN(outputFolder);

    // cleanup
    delete Line3D;
}
