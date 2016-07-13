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
