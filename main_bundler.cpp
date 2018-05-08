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
// This executable reads bundler results (bundle.rd.out) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the bundle file, you need to use the _original_
// (distorted) images!

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "image_folder", "folder containing the images (be carefull with the path if an image list is used!)", true, ".", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> bundleFileArg("b", "bundle_file", "full path to the bundle.*.out file (if not specified -> image_folder/../bundle.rd.out)", false, "", "string");
    cmd.add(bundleFileArg);

    TCLAP::ValueArg<std::string> imgListFileArg("f", "img_list", "full path to an optional image list (e.g. for the Dubrovnik6K dataset)", false, "", "string");
    cmd.add(imgListFileArg);

    TCLAP::ValueArg<std::string> extArg("t", "image_extension", "image extension (case sensitive), if not specified: jpg, png or bmp expected", false, "", "string");
    cmd.add(extArg);

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
    std::string imageFolder = inputArg.getValue().c_str();
    std::string bundleFile = bundleFileArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    std::string imageListFile = imgListFileArg.getValue().c_str();
    std::string imgExtension = extArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = imageFolder+"/Line3D++/";

    if(imgExtension.length() > 0 && imgExtension.substr(0,1) != ".")
        imgExtension = "."+imgExtension;

    if(bundleFile.length() == 0)
        bundleFile = imageFolder+"/../bundle.rd.out";

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

    // check if bundle.rd.out exists
    boost::filesystem::path bf(bundleFile);
    if(!boost::filesystem::exists(bf))
    {
        std::cerr << "bundle file '" << bundleFile << "' does not exist!" << std::endl;
        return -1;
    }

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU);

    // read bundle.rd.out
    std::ifstream bundle_file;
    bundle_file.open(bundleFile.c_str());

    std::string bundle_line;
    std::getline(bundle_file,bundle_line); // ignore first line...
    std::getline(bundle_file,bundle_line);

    // read number of images and 3D points
    std::stringstream bundle_stream(bundle_line);
    unsigned int num_cams,num_points;
    bundle_stream >> num_cams >> num_points;

    if(num_cams == 0 || num_points == 0)
    {
        std::cerr << "No cameras and/or points in bundle file!" << std::endl;
        return -1;
    }

    // read camera data (sequentially)
    std::vector<float> cams_focals(num_cams);
    std::vector<Eigen::Matrix3d> cams_rotation(num_cams);
    std::vector<Eigen::Vector3d> cams_translation(num_cams);
    std::vector<Eigen::Vector3d> cams_centers(num_cams);
    std::vector<std::pair<float,float> > cams_distortion(num_cams);
    for(unsigned int i=0; i<num_cams; ++i)
    {
        // focal_length,distortion
        double focal_length,dist1,dist2;
        std::getline(bundle_file,bundle_line);
        bundle_stream.str("");
        bundle_stream.clear();
        bundle_stream.str(bundle_line);
        bundle_stream >> focal_length >> dist1 >> dist2;

        cams_focals[i] = focal_length;
        cams_distortion[i] = std::pair<float,float>(dist1,dist2);

        // rotation
        Eigen::Matrix3d R;
        for(unsigned int j=0; j<3; ++j)
        {
            std::getline(bundle_file,bundle_line);
            bundle_stream.str("");
            bundle_stream.clear();
            bundle_stream.str(bundle_line);
            bundle_stream >> R(j,0) >> R(j,1) >> R(j,2);
        }

        // flip 2nd and 3rd line
        R(1,0) *= -1.0; R(1,1) *= -1.0; R(1,2) *= -1.0;
        R(2,0) *= -1.0; R(2,1) *= -1.0; R(2,2) *= -1.0;

        cams_rotation[i] = R;

        // translation
        std::getline(bundle_file,bundle_line);
        bundle_stream.str("");
        bundle_stream.clear();
        bundle_stream.str(bundle_line);
        Eigen::Vector3d t;
        bundle_stream >> t(0) >> t(1) >> t(2);

        // flip y and z!
        t(1) *= -1.0;
        t(2) *= -1.0;

        cams_translation[i] = t;

        // camera center
        Eigen::Matrix3d Rt = R.transpose();
        cams_centers[i] = Rt * (-1.0 * t);
    }

    // read features (for image similarity calculation)
    std::vector<std::list<unsigned int> > cams_worldpointIDs(num_cams);
    std::vector<std::vector<float> > cams_worldpointDepths(num_cams);
    for(unsigned int i=0; i<num_points; ++i)
    {
        // 3D position
        std::getline(bundle_file,bundle_line);
        std::istringstream iss_pos3D(bundle_line);
        double px,py,pz;
        iss_pos3D >> px >> py >> pz;
        Eigen::Vector3d pos3D(px,py,pz);

        // ignore color...
        std::getline(bundle_file,bundle_line);

        // view list
        std::getline(bundle_file,bundle_line);
        unsigned int num_views;

        std::istringstream iss(bundle_line);
        iss >> num_views;

        unsigned int camID,siftID;
        float posX,posY;
        for(unsigned int j=0; j<num_views; ++j)
        {
            iss >> camID >> siftID;
            iss >> posX >> posY;
            cams_worldpointIDs[camID].push_back(i);

            cams_worldpointDepths[camID].push_back((pos3D-cams_centers[camID]).norm());
        }
    }
    bundle_file.close();

    // check image list (if it exists)
    std::map<unsigned int,std::string> id2img;
    if(imageListFile.length() > 0)
    {
        std::ifstream img_list_file;
        img_list_file.open(imageListFile.c_str());

        std::string img_list_line;
        unsigned int id=0;
        while(std::getline(img_list_file,img_list_line))
        {
            std::stringstream img_list_stream(img_list_line);
            std::string fname,rest;
            img_list_stream >> fname >> rest;

            if(fname.length() > 0)
            {
                id2img[id] = fname;
            }

            ++id;
        }
    }

    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(int i=0; i<num_cams; ++i)
    {   
        // load image
        std::string img_filename = "";
        cv::Mat image;
        std::vector<boost::filesystem::wpath> possible_imgs;
        bool image_found = false;

        // check images from list
        if(id2img.find(i) != id2img.end())
        {
            image_found = true;
            img_filename = imageFolder+"/"+id2img[i];
        }

        if(!image_found)
        {
            // transform ID
            std::stringstream id_str;
            id_str << std::setfill('0') << std::setw(8) << i;
            std::string fixedID = id_str.str();

            if(imgExtension.length() == 0)
            {
                // look for common image extensions
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".jpg"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".JPG"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".png"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".PNG"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".jpeg"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".JPEG"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".bmp"));
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+".BMP"));
            }
            else
            {
                // use given extension
                possible_imgs.push_back(boost::filesystem::wpath(imageFolder+"/"+fixedID+imgExtension));
            }

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
        }

        if(image_found && cams_worldpointDepths[i].size() > 0)
        {
            // load image
            image = cv::imread(img_filename,CV_LOAD_IMAGE_GRAYSCALE);

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
            float d1 = cams_distortion[i].first;
            float d2 = cams_distortion[i].second;

            cv::Mat img_undist;
            if(fabs(d1) > L3D_EPS || fabs(d2) > L3D_EPS)
            {
                // undistorting
                Eigen::Vector3d radial(d1,d2,0.0);
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
                             cams_translation[i],med_depth,cams_worldpointIDs[i]);
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
