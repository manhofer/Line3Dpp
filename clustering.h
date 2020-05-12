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
 * Their code is put under the GNU GENERAL PUBLIC LICENSE.
 */

/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
