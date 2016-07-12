#ifndef I3D_LINE3D_PP_CLUSTERING_H_
#define I3D_LINE3D_PP_CLUSTERING_H_

/*
 * Disclaimer:
 * Re-implementation of the algorithm described in the paper
 *
 * Efficient Graph-based Image Segmentation,
 * P. Fezenszwalb, F. Huttenlocher,
 * International Journal of Computer Vision, 2004.
 *
 * and based on their official source code, which can be found at
 * from http://cs.brown.edu/~pff/segment/
 *
 * Their code is put under the GNU GENERAL PUBLIC LICENSE,
 * and so is this version.
 */

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

#include <list>
#include <algorithm>

#include "universe.h"

/**
 * Clustering
 * ====================
 * Clustering/Segmentation Algorithm.
 * [Felzenswalb & Huttenlocher, IJCV 2004]
 *
 * This code is an adaption of their original source code,
 * to fit into our datastructures.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    // edge in affinity matrix
    typedef struct {
        int i_;
        int j_;
        float w_;
    } CLEdge;

    // sorting function for edges
    static bool sortCLEdgesByWeight(const CLEdge& a, const CLEdge& b)
    {
        return a.w_ < b.w_;
    }

    // sort entries for sparse affinity matrix (CLEdges)
    static bool sortCLEdgesByCol(const CLEdge& a1, const CLEdge& a2)
    {
        return ((a1.j_ < a2.j_) || (a1.j_ == a2.j_ && a1.i_ < a2.i_));
    }

    static bool sortCLEdgesByRow(const CLEdge& a1, const CLEdge& a2)
    {
        return ((a1.i_ < a2.i_) || (a1.i_ == a2.i_ && a1.j_ < a2.j_));
    }

    // perform graph clustering
    // NOTE: edges are sorted during the process!
    CLUniverse* performClustering(std::list<CLEdge>& edges, int numNodes,
                                  float c);

}

#endif //I3D_LINE3D_PP_CLUSTERING_H_
