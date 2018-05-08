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

// INFO:
// This executable reads VisualSfM results (*.nvm) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the nvm file, you need to use the _original_
// (distorted) images!

int main(int argc, char *argv[])
{
    // Info: reads only the _first_ 3D model in the NVM file!

    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images (if not specified, path in .nvm file is expected to be correct)", false, "", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> nvmArg("m", "nvm_file", "full path to the VisualSfM result file (.nvm)", true, ".", "string");
    cmd.add(nvmArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> input_folder+'/Line3D++/')", false, "", "string");
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
    std::string inputFolder = inputArg.getValue().c_str();
    std::string nvmFile = nvmArg.getValue().c_str();

    // check if NVM file exists
    boost::filesystem::path nvm(nvmFile);
    if(!boost::filesystem::exists(nvm))
    {
        std::cerr << "NVM file " << nvmFile << " does not exist!" << std::endl;
        return -1;
    }

    bool use_full_image_path = false;
    if(inputFolder.length() == 0)
    {
        // parse input folder from .nvm file
        use_full_image_path = true;
        inputFolder = nvm.parent_path().string();
    }

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

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU);

    // read NVM file
    std::ifstream nvm_file;
    nvm_file.open(nvmFile.c_str());

    std::string nvm_line;
    std::getline(nvm_file,nvm_line); // ignore first line...
    std::getline(nvm_file,nvm_line); // ignore second line...

    // read number of images
    std::getline(nvm_file,nvm_line);
    std::stringstream nvm_stream(nvm_line);
    unsigned int num_cams;
    nvm_stream >> num_cams;

    if(num_cams == 0)
    {
        std::cerr << "No aligned cameras in NVM file!" << std::endl;
        return -1;
    }

    // read camera data (sequentially)
    std::vector<std::string> cams_imgFilenames(num_cams);
    std::vector<float> cams_focals(num_cams);
    std::vector<Eigen::Matrix3d> cams_rotation(num_cams);
    std::vector<Eigen::Vector3d> cams_translation(num_cams);
    std::vector<Eigen::Vector3d> cams_centers(num_cams);
    std::vector<float> cams_distortion(num_cams);
    for(unsigned int i=0; i<num_cams; ++i)
    {
        std::getline(nvm_file,nvm_line);

        // image filename
        std::string filename;

        // focal_length,quaternion,center,distortion
        double focal_length,qx,qy,qz,qw;
        double Cx,Cy,Cz,dist;

        nvm_stream.str("");
        nvm_stream.clear();
        nvm_stream.str(nvm_line);
        nvm_stream >> filename >> focal_length >> qw >> qx >> qy >> qz;
        nvm_stream >> Cx >> Cy >> Cz >> dist;

        cams_imgFilenames[i] = filename;
        cams_focals[i] = focal_length;
        cams_distortion[i] = dist;

        // rotation amd translation
        Eigen::Matrix3d R;
        R(0,0) = 1.0-2.0*qy*qy-2.0*qz*qz;
        R(0,1) = 2.0*qx*qy-2.0*qz*qw;
        R(0,2) = 2.0*qx*qz+2.0*qy*qw;

        R(1,0) = 2.0*qx*qy+2.0*qz*qw;
        R(1,1) = 1.0-2.0*qx*qx-2.0*qz*qz;
        R(1,2) = 2.0*qy*qz-2.0*qx*qw;

        R(2,0) = 2.0*qx*qz-2.0*qy*qw;
        R(2,1) = 2.0*qy*qz+2.0*qx*qw;
        R(2,2) = 1.0-2.0*qx*qx-2.0*qy*qy;

        Eigen::Vector3d C(Cx,Cy,Cz);
        cams_centers[i] = C;
        Eigen::Vector3d t = -R*C;

        cams_translation[i] = t;
        cams_rotation[i] = R;
    }

    // read number of images
    std::getline(nvm_file,nvm_line); // ignore line...
    std::getline(nvm_file,nvm_line);
    nvm_stream.str("");
    nvm_stream.clear();
    nvm_stream.str(nvm_line);
    unsigned int num_points;
    nvm_stream >> num_points;

    // read features (for image similarity calculation)
    std::vector<std::list<unsigned int> > cams_worldpointIDs(num_cams);
    std::vector<std::vector<float> > cams_worldpointDepths(num_cams);
    for(unsigned int i=0; i<num_points; ++i)
    {
        // 3D position
        std::getline(nvm_file,nvm_line);
        std::istringstream iss_point3D(nvm_line);
        double px,py,pz,colR,colG,colB;
        iss_point3D >> px >> py >> pz;
        iss_point3D >> colR >> colG >> colB;
        Eigen::Vector3d pos3D(px,py,pz);

        // measurements
        unsigned int num_views;
        iss_point3D >> num_views;

        unsigned int camID,siftID;
        float posX,posY;
        for(unsigned int j=0; j<num_views; ++j)
        {
            iss_point3D >> camID >> siftID;
            iss_point3D >> posX >> posY;
            cams_worldpointIDs[camID].push_back(i);

            cams_worldpointDepths[camID].push_back((pos3D-cams_centers[camID]).norm());
        }
    }
    nvm_file.close();

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<num_cams; ++i)
    {
        if(cams_worldpointDepths[i].size() > 0)
        {
            // parse filename
            std::string fname = cams_imgFilenames[i];
            boost::filesystem::path img_path(fname);

            // load image
            cv::Mat image;
            if(use_full_image_path)
                image = cv::imread(inputFolder+"/"+fname,CV_LOAD_IMAGE_GRAYSCALE);
            else
                image = cv::imread(inputFolder+"/"+img_path.filename().string(),CV_LOAD_IMAGE_GRAYSCALE);

            // setup intrinsics
            float px = float(image.cols)/2.0f;
            float py = float(image.rows)/2.0f;
            float f = cams_focals[i];

            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            K(0,0) = f;
            K(1,1) = f;
            K(0,2) = px;
            K(1,2) = py;
            K(2,2) = 1.0;

            // undistort (if necessary)
            float d = cams_distortion[i];

            cv::Mat img_undist;
            if(fabs(d) > L3D_EPS)
            {
                // undistorting
                Eigen::Vector3d radial(-d,0.0,0.0);
                Eigen::Vector2d tangential(0.0,0.0);
                Line3D->undistortImage(image,img_undist,radial,tangential,K);
            }
            else
            {
                // already undistorted
                img_undist = image;
            }

            // median point depth
            std::sort(cams_worldpointDepths[i].begin(),cams_worldpointDepths[i].end());
            size_t med_pos = cams_worldpointDepths[i].size()/2;
            float med_depth = cams_worldpointDepths[i].at(med_pos);

            // add to system
            Line3D->addImage(i,img_undist,K,cams_rotation[i],
                             cams_translation[i],
                             med_depth,cams_worldpointIDs[i]);
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
