#ifndef I3D_LINE3D_PP_CUDAWRAPPER_H_
#define I3D_LINE3D_PP_CUDAWRAPPER_H_

// check if CUDA available
#include "configLIBS.h"

#ifdef L3DPP_CUDA

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

// std
#include <list>
#include <vector>

// external
#include "cuda.h"
#include "cuda_runtime.h"
#include "math_constants.h"
#include "helper_math.h"
#include "boost/thread/mutex.hpp"

// internal
#include "dataArray.h"
#include "sparsematrix.h"
#include "commons.h"

/**
 * Line3D++ - CUDA functionality
 * ====================
 * All CUDA functions are defined here.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    // constants
    const unsigned int L3D_BLOCK_SIZE = 16;
    const int L3D_GPU_BUFFER_LEN = 3072;
    const int L3D_GPU_BUFFER_SIZE = L3D_GPU_BUFFER_LEN*L3D_GPU_BUFFER_LEN;

    // device constants
    __device__ const float L3D_EPS_GPU = 1e-12;
    __device__ const float L3D_PI_1_2_GPU = 1.571f;

    // feature matching
    extern unsigned int match_lines_GPU(L3DPP::DataArray<float4>* lines_src,
                                        L3DPP::DataArray<float4>* lines_tgt,
                                        L3DPP::DataArray<float>* F,
                                        L3DPP::DataArray<float>* RtKinv_src,
                                        L3DPP::DataArray<float>* RtKinv_tgt,
                                        const float3 C_src, const float3 C_tgt,
                                        std::vector<std::list<L3DPP::Match> >* matches,
                                        const unsigned int srcCamID, const unsigned int tgtCamID,
                                        const float epi_overlap, const int kNN);

    // scoring of matches
    extern void score_matches_GPU(L3DPP::DataArray<float4>* lines,
                                  L3DPP::DataArray<float4>* matches,
                                  L3DPP::DataArray<int2>* ranges,
                                  L3DPP::DataArray<float>* scores,
                                  L3DPP::DataArray<float2>* regularizers_tgt,
                                  L3DPP::DataArray<float>* RtKinv,
                                  const float3 C, const float two_sigA_sqr,
                                  const float k, const float min_similarity);

    // find collinear segments
    extern void find_collinear_segments_GPU(L3DPP::DataArray<char>* C,
                                            L3DPP::DataArray<float4>* lines,
                                            const float dist_t);

    // replicator dynamics diffusion [M.Donoser, BMVC'13]
    extern void replicator_dynamics_diffusion_GPU(L3DPP::SparseMatrix* &W, const std::string prefix);
}

#endif //L3DPP_CUDA

#endif //I3D_LINE3D_PP_CUDAWRAPPER_H_
