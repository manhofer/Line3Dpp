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
// This executable reads mavmap results (image-data-*.txt) and executes the Line3D++ algorithm.
// Currently, only the PINHOLE camera model is supported!
// If distortion coefficients are stored in the sfm_data file, you need to use the _original_
// (distorted) images!

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, ".", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> mavmapArg("b", "mavmap_output", "full path to the mavmap output (image-data-*.txt)", true, "", "string");
    cmd.add(mavmapArg);

    TCLAP::ValueArg<std::string> extArg("t", "image_extension", "image extension (case sensitive), if not specified: jpg, png or bmp expected", false, "", "string");
    cmd.add(extArg);

    TCLAP::ValueArg<std::string> prefixArg("f", "image_prefix", "optional image prefix", false, "", "string");
    cmd.add(prefixArg);

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
    std::string mavmapFile = mavmapArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    std::string imgExtension = extArg.getValue().c_str();
    std::string imgPrefix = prefixArg.getValue().c_str();
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

    if(imgExtension.substr(0,1) != ".")
        imgExtension = "."+imgExtension;

    // since no median depth can be computed without camera-to-worldpoint links
    // sigma_p MUST be positive and in pixels!
    if(sigmaP < L3D_EPS && constRegDepth < L3D_EPS)
    {
        std::cout << "sigma_p cannot be negative (i.e. in world coordiantes) when no valid regularization depth (--const_reg_depth) is given!" << std::endl;
        std::cout << "reverting to: sigma_p = 2.5px" << std::endl;
        sigmaP = 2.5f;
    }

    // check if mavmap file exists
    boost::filesystem::path bf(mavmapFile);
    if(!boost::filesystem::exists(bf))
    {
        std::cerr << "mavmap file '" << mavmapFile << "' does not exist!" << std::endl;
        return -1;
    }

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,false,useGPU);

    // read mavmap result
    std::ifstream mavmap_file;
    mavmap_file.open(mavmapFile.c_str());

    std::string mavmap_line;
    // check for comments...
    while(std::getline(mavmap_file,mavmap_line))
    {
        if(mavmap_line.substr(0,1) != "#")
            break;
    }

    // read camera data (sequentially)
    std::vector<std::string> cams_filenames;
    std::vector<std::pair<double,double> > cams_focals;
    std::vector<Eigen::Matrix3d> cams_rotation;
    std::vector<Eigen::Vector3d> cams_translation;
    std::vector<Eigen::Vector2d> cams_principle;
    do
    {
        if(mavmap_line.length() < 28)
            break;

        std::string filename,roll,pitch,yaw;
        std::string lat,lon,alt,h;
        std::string tx,ty,tz;
        std::string camID,camModel,fx,fy,cx,cy;

        std::stringstream mavmap_stream(mavmap_line);
        mavmap_stream >> filename >> roll >> pitch >> yaw;
        mavmap_stream >> lat >> lon >> alt >> h;
        mavmap_stream >> tx >> ty >> tz;
        mavmap_stream >> camID >> camModel >> fx >> fy >> cx >> cy;

        // check camera model
        if(camModel.substr(0,camModel.length()-1) != "PINHOLE")
        {
            std::cout << "only PINHOLE camera model supported..." << std::endl;
            return -1;
        }

        // filename
        cams_filenames.push_back(filename.substr(0,filename.length()-1));

        // rotation
        double r,y,p;
        std::stringstream  dat_stream(roll.substr(0,roll.length()-1));
        dat_stream >> r; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(pitch.substr(0,pitch.length()-1));
        dat_stream >> p; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(yaw.substr(0,yaw.length()-1));
        dat_stream >> y; dat_stream.str(""); dat_stream.clear();

        Eigen::Matrix3d R = Line3D->rotationFromRPY(r,p,y);

        // translation
        double Tx,Ty,Tz;
        dat_stream.str(tx.substr(0,tx.length()-1));
        dat_stream >> Tx; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(ty.substr(0,ty.length()-1));
        dat_stream >> Ty; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(tz.substr(0,tz.length()-1));
        dat_stream >> Tz; dat_stream.str(""); dat_stream.clear();

        Eigen::Vector3d t(Tx,Ty,Tz);

        // invert
        Eigen::MatrixXd Rt = Eigen::MatrixXd::Identity(4,4);
        Rt.block<3,3>(0,0) = R;
        Rt.block<3,1>(0,3) = t;
        Eigen::MatrixXd Rt_inv = Rt.inverse().eval().block<3, 4>(0,0);

        R = Rt_inv.block<3,3>(0,0);
        t = Rt_inv.block<3,1>(0,3);

        cams_rotation.push_back(R);
        cams_translation.push_back(t);

        // focal lengths
        double foc_x,foc_y;
        dat_stream.str(fx.substr(0,fx.length()-1));
        dat_stream >> foc_x; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(fy.substr(0,fy.length()-1));
        dat_stream >> foc_y; dat_stream.str(""); dat_stream.clear();

        cams_focals.push_back(std::pair<double,double>(foc_x,foc_y));

        // principle point
        double pp_x,pp_y;
        dat_stream.str(cx.substr(0,cx.length()-1));
        dat_stream >> pp_x; dat_stream.str(""); dat_stream.clear();
        dat_stream.str(cy.substr(0,cy.length()-1));
        dat_stream >> pp_y; dat_stream.str(""); dat_stream.clear();

        cams_principle.push_back(Eigen::Vector2d(pp_x,pp_y));

    }while(std::getline(mavmap_file,mavmap_line));
    mavmap_file.close();

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<cams_rotation.size(); ++i)
    {
        // load image
        std::string img_filename = imgPrefix+cams_filenames[i];
        cv::Mat image;
        std::vector<boost::filesystem::wpath> possible_imgs;

        if(imgExtension.length() == 0)
        {
            // look for common image extensions
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".jpg"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".JPG"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".png"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".PNG"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".jpeg"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".JPEG"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".bmp"));
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+".BMP"));
        }
        else
        {
            // use given extension
            possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/"+img_filename+imgExtension));
        }

        bool image_found = false;
        unsigned int pos = 0;
        while(!image_found && pos < possible_imgs.size())
        {
            if(boost::filesystem::exists(possible_imgs[pos]))
            {
                image_found = true;
                img_filename = possible_imgs[pos].string();
            }
            ++pos;
        }

        if(image_found)
        {
            // load image
            image = cv::imread(img_filename,CV_LOAD_IMAGE_GRAYSCALE);

            // setup intrinsics
            double px = cams_principle[i].x();
            double py = cams_principle[i].y();
            double fx = cams_focals[i].first;
            double fy = cams_focals[i].second;

            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            K(0,0) = fx;
            K(1,1) = fy;
            K(0,2) = px;
            K(1,2) = py;
            K(2,2) = 1.0;

            // set neighbors
            std::list<unsigned int> neighbor_list;
            size_t n_before = neighbors/2;
            for(int nID=int(i)-1; nID >= 0 && neighbor_list.size()<n_before; --nID)
            {
                neighbor_list.push_back(nID);
            }
            for(int nID=int(i)+1; nID < int(cams_rotation.size()) && neighbor_list.size() < neighbors; ++nID)
            {
                neighbor_list.push_back(nID);
            }

            // add to system
            Line3D->addImage(i,image,K,cams_rotation[i],
                             cams_translation[i],constRegDepth,neighbor_list);
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
