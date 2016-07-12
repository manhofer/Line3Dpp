#include "clustering.h"

namespace L3DPP
{
    //------------------------------------------------------------------------------
    CLUniverse* performClustering(std::list<CLEdge>& edges, int numNodes,
                                  float c)
    {
        if(edges.size() == 0)
            return NULL;

        // sort edges by weight (increasing)
        edges.sort(L3DPP::sortCLEdgesByWeight);

        // init universe
        CLUniverse *u = new CLUniverse(numNodes);

        // init thresholds
        float* threshold = new float[numNodes];
        for(int i=0; i < numNodes; ++i)
            threshold[i] = c;

        // perform clustering
        std::list<CLEdge>::const_iterator it = edges.begin();
        for(; it!=edges.end(); ++it)
        {
            CLEdge e = *it;

            // components conected by this edge
            int a = u->find(e.i_);
            int b = u->find(e.j_);
            if (a != b)
            {
                // not in the same segment yet
                if((e.w_ <= threshold[a]) && (e.w_ <= threshold[b]))
                {
                    // join nodes
                    u->join(a,b);
                    a = u->find(a);
                    threshold[a] = e.w_ + c/u->size(a);
                }
            }
        }

        // cleanup
        delete threshold;
        return u;
    }
}
