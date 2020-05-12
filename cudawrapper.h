#ifndef I3D_LINE3D_PP_CUDAWRAPPER_H_
#define I3D_LINE3D_PP_CUDAWRAPPER_H_

// check if CUDA available
#include "configLIBS.h"

#ifdef L3DPP_CUDA

/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
