#ifndef I3D_LINE3D_PP_UNIVERSE_H_
#define I3D_LINE3D_PP_UNIVERSE_H_

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

/**
 * Universe (for clustering.h)
 * ====================
 * Clustering/Segmentation Algorithm.
 * [Felzenswalb & Huttenlocher, IJCV 2004]
 *
 * This code is an adaption of their original source code,
 * to fit into our datastructures.
 * ====================
 * Author: M.Hofer, 2015
 */

// element of cluster universe
typedef struct {
    int rank_;
    int clusterID_;
    int size_;
} CLUnivElement;

namespace L3DPP
{
    // class that holds a clustering result
    class CLUniverse
    {
    public:
        CLUniverse(int numNodes) : num_(numNodes)
        {
            // init
            elements_ = new CLUnivElement[num_];
            for(int i = 0; i < num_; ++i)
            {
                elements_[i].rank_ = 0;
                elements_[i].size_ = 1;
                elements_[i].clusterID_ = i;
            }
        }

        ~CLUniverse()
        {
            delete [] elements_;
        }

        // find clusterID for given node
        int find(int nodeID)
        {
            int y = nodeID;
            while(y != elements_[y].clusterID_)
                y = elements_[y].clusterID_;

            elements_[nodeID].clusterID_ = y;
            return y;
        }

        // joins two nodes into one class/segment
        void join(int x, int y)
        {
            if(elements_[x].rank_ > elements_[y].rank_)
            {
                elements_[y].clusterID_ = x;
                elements_[x].size_ += elements_[y].size_;
            }
            else
            {
                elements_[x].clusterID_ = y;
                elements_[y].size_ += elements_[x].size_;
                if(elements_[x].rank_ == elements_[y].rank_)
                    elements_[y].rank_++;
            }
            --num_;
        }

        float size(int x) const { return elements_[x].size_; }
        int numSets() const { return num_; }

    private:
        CLUnivElement* elements_;
        int num_;
    };
}

#endif //I3D_LINE3D_PP_UNIVERSE_H_
