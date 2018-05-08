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
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

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

// INFO:
// This executable reads OpenMVG results (sfm_data.json) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the sfm_data file, you need to use the _original_
// (distorted) images!

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the original images", true, ".", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> jsonArg("j", "sfm_json_file", "full path to the OpenMVG result file (sfm_data.json)", true, ".", "string");
    cmd.add(jsonArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> input_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching (-1 --> use all)", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
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
    std::string inputFolder = inputArg.getValue().c_str();
    std::string jsonFile = jsonArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = inputFolder+"/Line3D++/";

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

    // check if json file exists
    boost::filesystem::path json(jsonFile);
    if(!boost::filesystem::exists(json))
    {
        std::cerr << "OpenMVG json file " << jsonFile << " does not exist!" << std::endl;
        return -1;
    }

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU);

    // parse json file
    std::ifstream jsonFileIFS(jsonFile.c_str());
    std::string str((std::istreambuf_iterator<char>(jsonFileIFS)),
                     std::istreambuf_iterator<char>());
    rapidjson::Document d;
    d.Parse(str.c_str());

    rapidjson::Value& s = d["views"];
    size_t num_cams =  s.Size();

    if(num_cams == 0)
    {
        std::cerr << "No aligned cameras in json file!" << std::endl;
        return -1;
    }

    // read image IDs and filename (sequentially)
    std::vector<std::string> cams_imgFilenames(num_cams);
    std::vector<unsigned int> cams_intrinsic_IDs(num_cams);
    std::vector<unsigned int> cams_view_IDs(num_cams);
    std::vector<unsigned int> cams_pose_IDs(num_cams);
    std::vector<bool> img_found(num_cams);
    std::map<unsigned int,unsigned int> pose2view;

    for(rapidjson::SizeType i=0; i<s.Size(); ++i)
    {
        rapidjson::Value& array_element = s[i];
        rapidjson::Value& view_data = array_element["value"]["ptr_wrapper"]["data"];

        std::string filename = view_data["filename"].GetString();
        unsigned int view_id = view_data["id_view"].GetUint();
        unsigned int intrinsic_id = view_data["id_intrinsic"].GetUint();
        unsigned int pose_id = view_data["id_pose"].GetUint();

        std::string full_path = inputFolder+"/"+filename;
        boost::filesystem::path full_path_check(full_path);
        if(boost::filesystem::exists(full_path_check))
        {
            // image exists
            cams_imgFilenames[i] = full_path;
            cams_view_IDs[i] = view_id;
            cams_intrinsic_IDs[i] = intrinsic_id;
            cams_pose_IDs[i] = pose_id;
            img_found[i] = true;

            pose2view[pose_id] = view_id;
        }
        else
        {
            // image not found...
            img_found[i] = false;
            std::cerr << "WARNING: image '" << filename << "' not found (ID=" << view_id << ")" << std::endl;
        }
    }

    // read intrinsics (sequentially)
    std::map<unsigned int,Eigen::Vector3d> radial_dist;
    std::map<unsigned int,Eigen::Vector2d> tangential_dist;
    std::map<unsigned int,Eigen::Matrix3d> K_matrices;
    std::map<unsigned int,bool> is_distorted;

    rapidjson::Value& intr = d["intrinsics"];
    size_t num_intrinsics =  intr.Size();

    if(num_intrinsics == 0)
    {
        std::cerr << "No intrinsics in json file!" << std::endl;
        return -1;
    }

    std::string cam_model;
    for(rapidjson::SizeType i=0; i<intr.Size(); ++i)
    {
        rapidjson::Value& array_element = intr[i];
        rapidjson::Value& intr_data = array_element["value"]["ptr_wrapper"]["data"];
        if (array_element["value"].HasMember("polymorphic_name"))
          cam_model = array_element["value"]["polymorphic_name"].GetString();
        unsigned int groupID = array_element["key"].GetUint();

        bool distorted = false;
        Eigen::Vector3d radial_d(0,0,0);
        Eigen::Vector2d tangential_d(0,0);

        double focal_length = intr_data["focal_length"].GetDouble();;

        Eigen::Vector2d principle_p;
        principle_p(0) = intr_data["principal_point"][0].GetDouble();
        principle_p(1) = intr_data["principal_point"][1].GetDouble();

        // check camera model for distortion
        if(cam_model.compare("pinhole_radial_k3") == 0)
        {
            // 3 radial
            radial_d(0) = intr_data["disto_k3"][0].GetDouble();
            radial_d(1) = intr_data["disto_k3"][1].GetDouble();
            radial_d(2) = intr_data["disto_k3"][2].GetDouble();
        }
        else if(cam_model.compare("pinhole_radial_k1") == 0)
        {
            // 1 radial
            radial_d(0) = intr_data["disto_k1"][0].GetDouble();
        }
        else if(cam_model.compare("pinhole_brown_t2") == 0)
        {
            // 3 radial
            radial_d(0) = intr_data["disto_t2"][0].GetDouble();
            radial_d(1) = intr_data["disto_t2"][1].GetDouble();
            radial_d(2) = intr_data["disto_t2"][2].GetDouble();
            // 2 tangential
            tangential_d(0) = intr_data["disto_t2"][3].GetDouble();
            tangential_d(1) = intr_data["disto_t2"][4].GetDouble();
        }
        else if(cam_model.compare("pinhole") != 0)
        {
            std::cerr << "WARNING: camera model '" << cam_model << "' for group " << groupID << " unknown! No distortion assumed..." << std::endl;
        }

        // check if distortion actually occured
        if(fabs(radial_d(0)) > L3D_EPS || fabs(radial_d(1)) > L3D_EPS || fabs(radial_d(2)) > L3D_EPS ||
                fabs(tangential_d(0)) > L3D_EPS || fabs(tangential_d(1)) > L3D_EPS)
        {
            distorted = true;
        }

        // create K
        Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
        K(0,0) = focal_length;
        K(1,1) = focal_length;
        K(0,2) = principle_p(0);
        K(1,2) = principle_p(1);
        K(2,2) = 1.0;

        // store
        radial_dist[groupID] = radial_d;
        tangential_dist[groupID] = tangential_d;
        K_matrices[groupID] = K;
        is_distorted[groupID] = distorted;
    }

    // read extrinsics (sequentially)
    std::map<unsigned int,Eigen::Vector3d> translations;
    std::map<unsigned int,Eigen::Vector3d> centers;
    std::map<unsigned int,Eigen::Matrix3d> rotations;

    rapidjson::Value& extr = d["extrinsics"];
    size_t num_extrinsics =  extr.Size();

    if(num_extrinsics == 0)
    {
        std::cerr << "No extrinsics in json file!" << std::endl;
        return -1;
    }

    for(rapidjson::SizeType i=0; i<extr.Size(); ++i)
    {
        rapidjson::Value& array_element = extr[i];
        unsigned int poseID = array_element["key"].GetUint();

        if(pose2view.find(poseID) != pose2view.end())
        {
            unsigned int viewID = pose2view[poseID];

            // rotation
            rapidjson::Value& _R = array_element["value"]["rotation"];
            Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
            R(0,0) = _R[0][0].GetDouble(); R(0,1) = _R[0][1].GetDouble(); R(0,2) = _R[0][2].GetDouble();
            R(1,0) = _R[1][0].GetDouble(); R(1,1) = _R[1][1].GetDouble(); R(1,2) = _R[1][2].GetDouble();
            R(2,0) = _R[2][0].GetDouble(); R(2,1) = _R[2][1].GetDouble(); R(2,2) = _R[2][2].GetDouble();

            // center
            rapidjson::Value& _C = array_element["value"]["center"];
            Eigen::Vector3d C;
            C(0) = _C[0].GetDouble(); C(1) = _C[1].GetDouble(); C(2) = _C[2].GetDouble();

            // translation
            Eigen::Vector3d t = -R*C;

            // store
            translations[viewID] = t;
            centers[viewID] = C;
            rotations[viewID] = R;
        }
        else
        {
            std::cerr << "WARNING: pose with ID " << poseID << " does not map to an image!" << std::endl;
        }
    }

    // read worldpoint data (sequentially)
    std::map<unsigned int,std::list<unsigned int> > views2wps;
    std::map<unsigned int,std::vector<float> > views2depths;

    rapidjson::Value& wps = d["structure"];
    size_t num_wps =  wps.Size();

    if(num_wps == 0)
    {
        std::cerr << "No worldpoints in json file!" << std::endl;
        return -1;
    }

    for(rapidjson::SizeType i=0; i<wps.Size(); ++i)
    {
        rapidjson::Value& array_element = wps[i];
        rapidjson::Value& wp_data = array_element["value"];

        // id and position
        unsigned int wpID = array_element["key"].GetUint();
        Eigen::Vector3d X;
        X(0) = wp_data["X"][0].GetDouble();
        X(1) = wp_data["X"][1].GetDouble();
        X(2) = wp_data["X"][2].GetDouble();

        // observations
        size_t num_obs = wp_data["observations"].Size();
        for(size_t j=0; j<num_obs; ++j)
        {
            unsigned int viewID = wp_data["observations"][j]["key"].GetUint();

            if(centers.find(viewID) != centers.end())
            {
                float depth = (centers[viewID]-X).norm();

                // store in list
                views2wps[viewID].push_back(wpID);
                views2depths[viewID].push_back(depth);
            }
        }
    }

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<num_cams; ++i)
    {
        unsigned int camID = cams_view_IDs[i];
        unsigned int intID = cams_intrinsic_IDs[i];

        if(views2wps.find(camID) != views2wps.end() && img_found[i] &&
                K_matrices.find(intID) != K_matrices.end())
        {
            // load image
            cv::Mat image = cv::imread(cams_imgFilenames[i],CV_LOAD_IMAGE_GRAYSCALE);

            // intrinsics
            Eigen::Matrix3d K = K_matrices[intID];

            // undistort (if necessary)
            bool distorted = is_distorted[intID];
            Eigen::Vector3d radial = radial_dist[intID];
            Eigen::Vector2d tangential = tangential_dist[intID];

            cv::Mat img_undist;
            if(distorted)
            {
                // undistorting
                Line3D->undistortImage(image,img_undist,radial,tangential,K);
            }
            else
            {
                // already undistorted
                img_undist = image;
            }

            // median point depth
            std::sort(views2depths[camID].begin(),views2depths[camID].end());
            size_t med_pos = views2depths[camID].size()/2;
            float med_depth = views2depths[camID].at(med_pos);

            // add to system
            Line3D->addImage(camID,img_undist,K,rotations[camID],
                             translations[camID],
                             med_depth,views2wps[camID]);
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
