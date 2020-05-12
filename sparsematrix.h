#ifndef I3D_LINE3D_PP_SPARSEMATRIX_H_
#define I3D_LINE3D_PP_SPARSEMATRIX_H_

/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// check for CUDA
#include "configLIBS.h"

#ifdef L3DPP_CUDA

// internal
#include "clustering.h"
#include "dataArray.h"

/**
 * Line3D++ - Sparsematrix
 * ====================
 * Sparse GPU matrix.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    class SparseMatrix
    {
    public:
        SparseMatrix(std::list<L3DPP::CLEdge>& entries, const unsigned int num_rows_cols,
                     const float normalization_factor=1.0f,
                     const bool sort_by_row=false, const bool already_sorted=false);
        SparseMatrix(SparseMatrix* M, const bool change_sorting=false);
        ~SparseMatrix();

        // check element sorting
        bool isRowSorted(){
            return row_sorted_;
        }
        bool isColSorted(){
            return !row_sorted_;
        }

        // data access
        unsigned int num_rows_cols(){
            return num_rows_cols_;
        }
        unsigned int num_entries(){
            return num_entries_;
        }

        // CPU/GPU data
        L3DPP::DataArray<float4>* entries(){
            return entries_;
        }
        L3DPP::DataArray<int>* start_indices(){
            return start_indices_;
        }

        // download entries to CPU
        void download(){
            entries_->download();
        }

    private:
        // CPU/GPU data
        L3DPP::DataArray<float4>* entries_;
        L3DPP::DataArray<int>* start_indices_;

        bool row_sorted_;
        unsigned int num_rows_cols_;
        unsigned int num_entries_;
    };
}

#endif //L3DPP_CUDA

#endif //I3D_LINE3D_PP_SPARSEMATRIX_H_
