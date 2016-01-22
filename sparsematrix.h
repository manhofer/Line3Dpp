#ifndef I3D_LINE3D_PP_SPARSEMATRIX_H_
#define I3D_LINE3D_PP_SPARSEMATRIX_H_

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
