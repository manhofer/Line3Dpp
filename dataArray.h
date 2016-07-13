#ifndef I3D_LINE3D_PP_DATAARRAY_H_
#define I3D_LINE3D_PP_DATAARRAY_H_

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

// internal
#include "configLIBS.h"

// external
#ifdef L3DPP_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#endif //L3DPP_CUDA

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/array.hpp>

// std
#include <vector>
#include <iostream>

/**
 * Line3D++ - DataArray CPU/GPU
 * ====================
 * DataArray that can be moved from CPU to GPU
 * and vice versa. Adapted from the ImageUtilities lib
 * by Manuel Werlberger.
 * ====================
 * Author: M.Hofer, 2015
 */

#ifndef L3DPP_CUDA
    // float4 - for non-CUDA machines
    struct float4
    {
        float x;
        float y;
        float z;
        float w;
    };

    // float2 - for non-CUDA machines
    struct float2
    {
        float x;
        float y;
    };
#endif //L3DPP_CUDA

// define serialization for float2/4
namespace boost
{
    namespace serialization
    {
        template<class Archive>
        void serialize(Archive & ar, float4& f, const unsigned int version)
        {
          ar & boost::serialization::make_nvp("x", f.x);
          ar & boost::serialization::make_nvp("y", f.y);
          ar & boost::serialization::make_nvp("z", f.z);
          ar & boost::serialization::make_nvp("w", f.w);
        }

        template<class Archive>
        void serialize(Archive & ar, float2& f, const unsigned int version)
        {
          ar & boost::serialization::make_nvp("x", f.x);
          ar & boost::serialization::make_nvp("y", f.y);
        }
    }
}

namespace L3DPP
{
    // floatN array (2D)
    template <class PixelType>
    class DataArray
    {
    public:
        DataArray()
        {
            width_ = 0;
            height_ = 0;
            real_width_ = 0;

            // CPU
            pitchCPU_ = 0;
            strideCPU_ = 0;

            // GPU
            dataGPU_ = NULL;
            pitchGPU_ = 0;
            strideGPU_ = 0;
        }

        DataArray(unsigned int width,
                  unsigned int height,
                  const bool allocate_GPU_memory=false,
                  const std::vector<PixelType>& data=std::vector<PixelType>()) :
            width_(width), height_(height)
        {
            // pitch (CPU)
            pitchCPU_ = width_*sizeof(PixelType);

            unsigned int elements2pitch;
            if(pitchCPU_ % 32 == 0)
                elements2pitch = 0;
            else
                elements2pitch = (32-(pitchCPU_ % 32))/sizeof(PixelType);

            width += elements2pitch;
            pitchCPU_ = width*sizeof(PixelType);
            strideCPU_ = pitchCPU_/sizeof(PixelType);
            real_width_ = width;

            // CPU --> stored line by line
            dataCPU_ = std::vector<PixelType>(width*height_);
            if(data.size() == width_*height_)
            {
                for(unsigned int h=0; h<height_; ++h)
                    for(unsigned int w=0; w<width_; ++w)
                        dataCPU(w,h)[0] = data[h*width_+w];
            }

            // GPU
            dataGPU_ = NULL;
            pitchGPU_ = 0;
            strideGPU_ = 0;

#ifdef L3DPP_CUDA
            if(allocate_GPU_memory)
            {
                allocateGPU();

                if(data.size() == width_*height_)
                {
                    upload();
                }
            }
#endif //L3DPP_CUDA
        }

        ~DataArray()
        {
#ifdef L3DPP_CUDA
            // delete GPU data
            removeFromGPU();
#endif //L3DPP_CUDA
        }

        // data access
        PixelType* dataCPU(unsigned int x=0,
                           unsigned int y=0){
            if(dataCPU_.size() > 0 && x < width_ && y < height_)
                return &dataCPU_[y*strideCPU_+x];
            else
                return NULL;
        }

#ifdef L3DPP_CUDA
        PixelType* dataGPU(unsigned int x=0,
                           unsigned int y=0)
        {
            if(dataGPU_ == NULL)
            {
                std::cerr << "DataArray::dataGPU(): data is _not_ on GPU!" << std::endl;
                return NULL;
            }

            if(dataGPU_ != NULL && x < width_ && y < height_)
                return &dataGPU_[y*strideGPU_+x];
            else
                return NULL;
        }

        // data transfer CPU/GPU
        void upload(const bool allocate_only=false)
        {
            if(dataGPU_ == NULL)
                allocateGPU();

            // host -> device
            if(dataGPU_ != NULL)
            {
                cudaError_t status = cudaMemcpy2D(dataGPU_,pitchGPU_,
                                                  &dataCPU_[0],pitchCPU_,
                                                  width_*sizeof(PixelType),height_,
                                                  cudaMemcpyHostToDevice);

                if(status != cudaSuccess)
                {
                    std::cerr << "DataArray::upload(): copying from CPU to GPU failed...[" << cudaGetErrorString(status) << "]" << std::endl;
                }
            }
            else
            {
               std::cerr << "DataArray::upload(): GPU memory not allocated..." << std::endl;
            }

            cudaDeviceSynchronize();
        }

        void download()
        {
            // device -> host
            if(dataGPU_ != NULL)
            {
                cudaError_t status = cudaMemcpy2D(&dataCPU_[0],pitchCPU_,
                                                  dataGPU_,pitchGPU_,
                                                  width_*sizeof(PixelType),height_,
                                                  cudaMemcpyDeviceToHost);

                if(status != cudaSuccess)
                {
                    std::cerr << "DataArray::download(): copying from GPU to CPU failed... [" << cudaGetErrorString(status) << "]" << std::endl;
                }
            }
        }

        void removeFromGPU()
        {
            if(dataGPU_ != NULL)
            {
                cudaError_t status = cudaFree((void *)dataGPU_);

                if(status != cudaSuccess)
                {
                    std::cerr << "DataArray::removeFromGPU(): could not remove data from GPU...[" << cudaGetErrorString(status) << "]" << std::endl;
                }

                dataGPU_ = NULL;
                pitchGPU_ = 0;
                strideGPU_ = 0;
            }
        }
#endif //L3DPP_CUDA

        // set constant value (CPU only!)
        void setValue(const PixelType p, const bool uploadToGPU=false)
        {
            for(unsigned int i=0; i<dataCPU_.size(); ++i)
                dataCPU_[i] = p;

#ifdef L3DPP_CUDA
            if(uploadToGPU)
                upload();
#endif //L3DPP_CUDA
        }

        // copy function (cpu tp cpu!)
        void copyTo(DataArray* dst, const bool uploadToGPU=false)
        {
            // cpu --> cpu
            for(unsigned int h=0; h<height_; ++h)
                for(unsigned int w=0; w<width_; ++w)
                    dst->dataCPU(w,h)[0] = dataCPU(w,h)[0];

#ifdef L3DPP_CUDA
            if(uploadToGPU)
                dst->upload();
#endif //L3DPP_CUDA
        }

        // basics
        unsigned int width() const {return width_;}
        unsigned int real_width() const {return real_width_;}
        unsigned int height() const {return height_;}
        size_t pitchCPU() const {return pitchCPU_;}
        size_t strideCPU() const {return strideCPU_;}

#ifdef L3DPP_CUDA
        size_t pitchGPU() const
        {
            if(dataGPU_ == NULL)
            {
                std::cerr << "DataArray::pitchGPU(): data is _not_ on GPU!" << std::endl;
            }

            return pitchGPU_;
        }
        size_t strideGPU() const
        {
            if(dataGPU_ == NULL)
            {
                std::cerr << "DataArray::strideGPU(): data is _not_ on GPU!" << std::endl;
            }

            return strideGPU_;
        }
        bool onGPU() const {return (dataGPU_ != NULL);}
#endif //L3DPP_CUDA
        size_t bytes() const {return height_*pitchCPU_;}

    private:

#ifdef L3DPP_CUDA
        // allocate GPU memory
        void allocateGPU()
        {
            if(dataGPU_ != NULL)
                return;

            if(width_ > 0 && height_ > 0)
            {
                dataGPU_ = 0;
                cudaError_t status = cudaMallocPitch((void **)&dataGPU_, &pitchGPU_,
                                                     width_*sizeof(PixelType), height_);

                if(status != cudaSuccess)
                {
                    std::cerr << "DataArray::allocateGPU(): GPU memory could not be allocated...[" << cudaGetErrorString(status) << "]" << std::endl;
                    dataGPU_ = NULL;
                    pitchGPU_ = 0;
                    strideGPU_ = 0;
                    return;
                }

                strideGPU_ = pitchGPU_/sizeof(PixelType);
            }
            else
            {
                std::cerr << "DataArray::allocateGPU(): width or height are zero! w=" << width_ << " h=" << height_ << std::endl;
            }

            cudaDeviceSynchronize();
        }
#endif //L3DPP_CUDA

        // basic
        unsigned int width_;
        unsigned int height_;
        unsigned int real_width_;

        // CPU
        std::vector<PixelType> dataCPU_;
        size_t pitchCPU_;
        size_t strideCPU_;

        // GPU
        PixelType* dataGPU_;
        size_t pitchGPU_;
        size_t strideGPU_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("width_", width_);
            ar & boost::serialization::make_nvp("height_", height_);
            ar & boost::serialization::make_nvp("real_width_", real_width_);
            ar & boost::serialization::make_nvp("pitchCPU_", pitchCPU_);
            ar & boost::serialization::make_nvp("strideCPU_", strideCPU_);
            ar & boost::serialization::make_nvp("pitchGPU_", pitchGPU_);
            ar & boost::serialization::make_nvp("strideGPU_", strideGPU_);

            if(Archive::is_loading::value)
            {
                dataGPU_ = NULL;
                pitchGPU_ = 0;
                strideGPU_ = 0;
                dataCPU_ = std::vector<PixelType>(real_width_*height_);
            }

            ar & boost::serialization::make_array<PixelType>(&dataCPU_[0],dataCPU_.size());
        }
    };
}

#endif //I3D_LINE3D_PP_DATAARRAY_H_
