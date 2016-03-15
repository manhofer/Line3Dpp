#include "cudawrapper.h"

#ifdef L3DPP_CUDA

namespace L3DPP
{
////////////////////////////////////////////////////////////////////////////////
// helper function for rounded-up division
int divUp(int a, int b)
{
    float res = float(a)/float(b);
    return ceil(res);
}

////////////////////////////////////////////////////////////////////////////////
// DEVICE
////////////////////////////////////////////////////////////////////////////////
__device__ float3 D_normalize_hom_coords_2D(float3 p)
{
    if(fabs(p.z) > L3D_EPS_GPU)
    {
        p /= p.z;
        p.z = 1;
        return p;
    }
    else
    {
        return make_float3(0,0,0);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Note: point needs to be normalized! (--> p.z == 1)
__device__ float D_distance_p2l_2D_f3(const float3 line, const float3 p)
{
    return fabs((line.x*p.x+line.y*p.y+line.z)/sqrtf(line.x*line.x+line.y*line.y));
}

////////////////////////////////////////////////////////////////////////////////
__device__ float3 D_line_direction_3D(const float3 P1, const float3 P2)
{
    return normalize(P2-P1);
}

////////////////////////////////////////////////////////////////////////////////
__device__ float D_undirected_angle_3D_DEG(const float3 v1, const float3 v2)
{
    float angle = acos(fmax(fmin(dot(v1,v2),1.0f),-1.0f))/CUDART_PI*180.0f;
    if(angle > 90.0f)
        angle = 180.0f-angle;

    return angle;
}

////////////////////////////////////////////////////////////////////////////////
__device__ float3 D_mult_matrix_vector_3(const float3 vec, const float* mat,
                                         const int stride, const bool transpose)
{
    float _in[3],_out[3];
    _in[0] = vec.x; _in[1] = vec.y; _in[2] = vec.z;
    _out[0] = 0.0f; _out[1] = 0.0f; _out[2] = 0.0f;

    for(int r=0; r<3; ++r)
    {
        for(int c=0; c<3; ++c)
        {
            if(!transpose)
                _out[r] += mat[r*stride+c]*_in[c];
            else
                _out[r] += mat[c*stride+r]*_in[c];
        }
    }

    return make_float3(_out[0],_out[1],_out[2]);
}

////////////////////////////////////////////////////////////////////////////////
// Note: points needs to be normalized! (--> p.z == 1),
// q needs to be collinear with p1 and p2!
__device__ bool D_point_on_segment_2D_f3(const float3 p1, const float3 p2,
                                         const float3 q)
{
    float2 v1 = make_float2(p1.x-q.x,p1.y-q.y);
    float2 v2 = make_float2(p2.x-q.x,p2.y-q.y);
    return (dot(v1,v2) < L3D_EPS_GPU);
}

////////////////////////////////////////////////////////////////////////////////
__device__ float D_segment_overlap_2D(const float3 src_p1, const float3 src_p2,
                                      const float3 proj_q1, const float3 proj_q2)
{
    // points are supposed to be collinear!
    float len_src = length(src_p1-src_p2);
    float len_tgt = length(proj_q1-proj_q2);

    if(len_src < 1.0f || len_tgt < 1.0f)
        return 0.0f;

    if(D_point_on_segment_2D_f3(src_p1,src_p2,proj_q1) &&
       D_point_on_segment_2D_f3(src_p1,src_p2,proj_q2))
    {
        // both target points within the ref segment
        return len_tgt/len_src;
    }
    else if(D_point_on_segment_2D_f3(proj_q1,proj_q2,src_p1) &&
            D_point_on_segment_2D_f3(proj_q1,proj_q2,src_p2))
    {
        // both source points within the tgt segment
        return len_src/len_tgt;
    }
    else if(D_point_on_segment_2D_f3(src_p1,src_p2,proj_q1))
    {
        float len1 = length(src_p2-proj_q2);
        float len2 = length(src_p1-proj_q2);

        // overlap exists
        if(D_point_on_segment_2D_f3(proj_q1,proj_q2,src_p1) && len1 > 1.0f)
            return length(proj_q1-src_p1)/len1;
        else if(len2 > 1.0f)
            return length(proj_q1-src_p2)/len2;
    }
    else if(D_point_on_segment_2D_f3(src_p1,src_p2,proj_q2))
    {
        float len1 = length(src_p1-proj_q1);
        float len2 = length(src_p2-proj_q1);

        // overlap exists
        if(D_point_on_segment_2D_f3(proj_q1,proj_q2,src_p2) && len1 > 1.0f)
            return length(proj_q2-src_p2)/len1;
        else if(len2 > 1.0f)
            return length(proj_q2-src_p1)/len2;
    }

    // no overlap
    return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
__device__ float2 D_triangulate_depth(const float3 p1, const float3 p2,
                                      const float3 q1, const float3 q2,
                                      const float3 C_src, const float3 C_tgt,
                                      const float* RtKinv_src, const float* RtKinv_tgt,
                                      const int stride)
{
    float2 d = make_float2(-1,-1);

    // point rays
    float3 ray_p1 = normalize(D_mult_matrix_vector_3(p1,RtKinv_src,stride,false));
    float3 ray_p2 = normalize(D_mult_matrix_vector_3(p2,RtKinv_src,stride,false));
    float3 ray_q1 = normalize(D_mult_matrix_vector_3(q1,RtKinv_tgt,stride,false));
    float3 ray_q2 = normalize(D_mult_matrix_vector_3(q2,RtKinv_tgt,stride,false));

    // plane
    float3 n = normalize(cross(ray_q1,ray_q2));

    float dotp1 = dot(n,ray_p1);
    float dotp2 = dot(n,ray_p2);
    if(fabs(dotp1) < L3D_EPS_GPU || fabs(dotp2) < L3D_EPS_GPU)
        return d;

    float d1 = (dot(C_tgt,n) - dot(n,C_src)) / dotp1;
    float d2 = (dot(C_tgt,n) - dot(n,C_src)) / dotp2;
    return make_float2(d1,d2);
}

////////////////////////////////////////////////////////////////////////////////
__device__ float3 D_unproject(const float3 p1, const float* RtKinv, const int stride,
                              const float3 C, const float depth)
{
    return C + depth*normalize(D_mult_matrix_vector_3(p1,RtKinv,stride,false));
}

////////////////////////////////////////////////////////////////////////////////
__device__ float D_smaller_angle(const float2 v1, const float2 v2)
{
    float ang = acos(fmax(fmin(dot(v1,v2),1.0f),-1.0f));
    if(ang > CUDART_PIO2)
        ang = CUDART_PI-ang;

    return ang;
}

////////////////////////////////////////////////////////////////////////////////
// KERNELS
////////////////////////////////////////////////////////////////////////////////
__global__ void K_match_lines(const int width, const int height,
                              const int offset_src, float4* buffer, const int stride,
                              float* overlaps, const int o_stride,
                              const float4* lines_src, const float4* lines_tgt,
                              const float* F, const float* RtKinv_src,
                              const float* RtKinv_tgt, const int rf_stride,
                              const float3 C_src, const float3 C_tgt,
                              const float epi_overlap)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        float4 result = make_float4(-1,-1,-1,-1);

        // src
        int srcID = y+offset_src;
        float4 l_src_data = lines_src[srcID];
        float3 p1 = make_float3(l_src_data.x,l_src_data.y,1.0f);
        float3 p2 = make_float3(l_src_data.z,l_src_data.w,1.0f);

        // tgt
        int tgtID = x;
        float4 l_tgt_data = lines_tgt[tgtID];
        float3 q1 = make_float3(l_tgt_data.x,l_tgt_data.y,1.0f);
        float3 q2 = make_float3(l_tgt_data.z,l_tgt_data.w,1.0f);
        float3 l_tgt = cross(q1,q2);

        // epipolar lines
        float3 epi_p1 = D_mult_matrix_vector_3(p1,F,rf_stride,false);
        float3 epi_p2 = D_mult_matrix_vector_3(p2,F,rf_stride,false);

        // intersect
        float3 l2_p1 = D_normalize_hom_coords_2D(cross(l_tgt,epi_p1));
        float3 l2_p2 = D_normalize_hom_coords_2D(cross(l_tgt,epi_p2));

        if(int(l2_p1.z) == 0 || int(l2_p2.z) == 0)
        {
            // intersections not valid
            buffer[y*stride+x] = result;
            overlaps[y*o_stride+x] = 0.0f;
            return;
        }

        // check for overlap
        float overlap = D_segment_overlap_2D(q1,q2,l2_p1,l2_p2);

        if(overlap > epi_overlap)
        {
            // compute depths
            float2 depths1 = D_triangulate_depth(p1,p2,q1,q2,C_src,C_tgt,
                                                 RtKinv_src,RtKinv_tgt,
                                                 rf_stride);
            float2 depths2 = D_triangulate_depth(q1,q2,p1,p2,C_tgt,C_src,
                                                 RtKinv_tgt,RtKinv_src,
                                                 rf_stride);

            result.x = depths1.x;
            result.y = depths1.y;
            result.z = depths2.x;
            result.w = depths2.y;
        }

        buffer[y*stride+x] = result;
        overlaps[y*o_stride+x] = overlap;
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void K_score_matches(const int num_matches, const float4* lines,
                                const float4* matches, float* scores,
                                const int2* ranges, const float2* reg_tgt,
                                const float* RtKinv, const int r_stride,
                                const float3 C, const float angle_reg,
                                const float k, const float sim_t)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < num_matches && y < 1)
    {
        // src match data
        float4 m = matches[x];

        // src target_cam
        int tgt_cam_src = m.y;

        // src depths
        float d1_src = m.z;
        float d2_src = m.w;

        // line points
        int lID = m.x;
        float4 l_src_data = lines[lID];
        float3 p1 = make_float3(l_src_data.x,l_src_data.y,1.0f);
        float3 p2 = make_float3(l_src_data.z,l_src_data.w,1.0f);

        // src dir
        float3 P1 = D_unproject(p1,RtKinv,r_stride,C,d1_src);
        float3 P2 = D_unproject(p2,RtKinv,r_stride,C,d2_src);
        float3 dir_src = D_line_direction_3D(P1,P2);

        // position regularizers
        float pos_reg1,pos_reg2;
        float sig1 = k*d1_src;
        float sig2 = k*d2_src;
        pos_reg1 = 2.0f*sig1*sig1;
        pos_reg2 = 2.0f*sig2*sig2;

        // position regularizers (tgt)
        float pos_reg1_tgt = 2.0f*reg_tgt[x].x*reg_tgt[x].x;
        float pos_reg2_tgt = 2.0f*reg_tgt[x].y*reg_tgt[x].y;

        // average regularizer
        pos_reg1 = 0.5f*(pos_reg1+pos_reg1_tgt);
        pos_reg2 = 0.5f*(pos_reg2+pos_reg2_tgt);

        // ranges
        int start = ranges[lID].x;
        int end = ranges[lID].y;

        float score3D = 0.0f;
        int current_cam = -1;
        float current_max_sim = 0.0f;
        for(int i=start; i<=end; ++i)
        {
            // tgt match data
            float4 m2 = matches[i];

            // src target_cam
            int tgt_cam_tgt = m2.y;

            if(tgt_cam_src != tgt_cam_tgt)
            {
                // tgt depths
                float d1_tgt = m2.z;
                float d2_tgt = m2.w;

                // tgt dir
                float3 Q1 = D_unproject(p1,RtKinv,r_stride,C,d1_tgt);
                float3 Q2 = D_unproject(p2,RtKinv,r_stride,C,d2_tgt);
                float3 dir_tgt = D_line_direction_3D(Q1,Q2);

                // angular similarity
                float angle = D_undirected_angle_3D_DEG(dir_src,dir_tgt);
                float sim_a = expf(-angle*angle/angle_reg);

                // position similarity
                float d1 = d1_src-d1_tgt;
                float d2 = d2_src-d2_tgt;

                float sim_p1 = expf(-d1*d1/pos_reg1);
                float sim_p2 = expf(-d2*d2/pos_reg2);
                float sim_p = fmin(sim_p1,sim_p2);

                // total similarity
                float sim = fmin(sim_a,sim_p);

                // truncate
                if(sim < sim_t)
                    sim = 0.0f;

                // update current max score
                current_max_sim = fmax(current_max_sim,sim);

                if(current_cam != tgt_cam_tgt)
                {
                    // new target cam
                    score3D += current_max_sim;
                    current_max_sim = 0.0f;
                    current_cam = tgt_cam_tgt;
                }
            }
        }

        // final update
        score3D += current_max_sim;

        scores[x] = score3D;
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void K_collinearity(const float4* lines, char* C, const int stride,
                               const int size, const float dist_t)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < size && y < size && x >= y)
    {
        if(x == y)
        {
            // no self-collinearity
            C[y*stride+x] = 0;
        }
        else
        {
            // line data
            float4 l1 = lines[x];
            float4 l2 = lines[y];

            float3 p[2],q[2];
            p[0] = make_float3(l1.x,l1.y,1.0f);
            p[1] = make_float3(l1.z,l1.w,1.0f);
            q[0] = make_float3(l2.x,l2.y,1.0f);
            q[1] = make_float3(l2.z,l2.w,1.0f);

            // check location (overlap)
            if(D_point_on_segment_2D_f3(p[0],p[1],q[0]) ||
                    D_point_on_segment_2D_f3(p[0],p[1],q[1]) ||
                    D_point_on_segment_2D_f3(q[0],q[1],p[0]) ||
                    D_point_on_segment_2D_f3(q[0],q[1],p[1]))
            {
                // overlap -> not collinear
                C[y*stride+x] = 0;
                C[x*stride+y] = 0;
                return;
            }

            // define line
            float3 line1 = cross(p[0],p[1]);
            float3 line2 = cross(q[0],q[1]);

            // compute distances
            float d1 = fmax(D_distance_p2l_2D_f3(line1,q[0]),
                            D_distance_p2l_2D_f3(line1,q[1]));
            float d2 = fmax(D_distance_p2l_2D_f3(line2,p[0]),
                            D_distance_p2l_2D_f3(line2,p[1]));

            if(fmax(d1,d2) < dist_t)
            {
                C[y*stride+x] = 1;
                C[x*stride+y] = 1;
            }
            else
            {
                C[y*stride+x] = 0;
                C[x*stride+y] = 0;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void K_sparseMat_row_normalization(float4* data, const int* start_indices,
                                              const int num_rows, const int num_entries)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x == 0 && y < num_rows)
    {
        int start = start_indices[y];

        if(y >= 0)
        {
            // compute sum
            float sum = 0.0f;
            int i = start;
            while(i < num_entries)
            {
                float4 e = data[i];
                int row = e.x;

                if(row != y)
                    break;

                sum += e.z;
                ++i;
            }

            // check for precision errors
            if(sum < L3D_EPS_GPU)
                sum = L3D_EPS_GPU;

            // normalize
            i = start;
            while(i < num_entries)
            {
                int row = data[i].x;

                if(row != y)
                    break;

                data[i].z /= sum;
                ++i;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void K_sparseMat_diffusion_step(const float4* P, const float4* W,
                                           const int* P_rows, const int* W_cols,
                                           float4* P_prime, const int* P_prime_rows,
                                           const int num_entries)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x == 0 && y < num_entries)
    {
        // get data
        float4 data = P[y];

        // transpose
        int r = data.y;
        int c = data.x;

        // row[P]*col[W]
        float mul = 0.0f;
        int start_P = P_rows[r];
        int start_W = W_cols[c];
        while(start_P < num_entries && start_W < num_entries)
        {
            float4 d1 = P[start_P];
            float4 d2 = W[start_W];

            int row1 = d1.x;
            int col2 = d2.y;

            if(row1 != r || col2 != c)
                break;

            mul += (d1.z*d2.z);
            ++start_P;
            ++start_W;
        }

        // multiply with transposed
        mul *= data.z;

        if(mul < L3D_EPS_GPU)
            mul = L3D_EPS_GPU;

        // store
        int s = P_prime_rows[r];
        bool found = false;
        while(s < num_entries && !found)
        {
            float4 dat = P_prime[s];
            int row = dat.x;
            int col = dat.y;

            if(row != r)
                break;

            if(col == c)
            {
                P_prime[s].z = mul;
                found = true;
            }

            ++s;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
unsigned int match_lines_GPU(L3DPP::DataArray<float4>* lines_src,
                             L3DPP::DataArray<float4>* lines_tgt,
                             L3DPP::DataArray<float>* F,
                             L3DPP::DataArray<float>* RtKinv_src,
                             L3DPP::DataArray<float>* RtKinv_tgt,
                             const float3 C_src, const float3 C_tgt,
                             std::vector<std::list<L3DPP::Match> >* matches,
                             const unsigned int srcCamID, const unsigned int tgtCamID,
                             const float epi_overlap, const int kNN)
{
    // init
    unsigned int block_size = L3D_BLOCK_SIZE;
    int width = lines_tgt->width();
    int height = lines_src->width();
    boost::mutex match_mutex;
    unsigned int num_matches = 0;

    // define grid
    dim3 dimBlock = dim3(block_size,block_size);

    // matching data
    int buffer_h = std::min(height,std::max(int(L3D_GPU_BUFFER_SIZE/width),1));
    L3DPP::DataArray<float4>* buffer = new L3DPP::DataArray<float4>(width,buffer_h,true);
    L3DPP::DataArray<float>* overlaps = new L3DPP::DataArray<float>(width,buffer_h,true);

    for(int offset_h = 0; offset_h < height; offset_h += buffer_h)
    {
        int current_height = std::min(buffer_h,height-offset_h);
        dim3 dimGrid = dim3(divUp(width, dimBlock.x),
                            divUp(current_height, dimBlock.y));

        L3DPP::K_match_lines <<< dimGrid, dimBlock >>> (width,current_height,offset_h,buffer->dataGPU(),
                                                        buffer->strideGPU(),overlaps->dataGPU(),
                                                        overlaps->strideGPU(),lines_src->dataGPU(),
                                                        lines_tgt->dataGPU(),
                                                        F->dataGPU(),RtKinv_src->dataGPU(),
                                                        RtKinv_tgt->dataGPU(),F->strideGPU(),
                                                        C_src,C_tgt,epi_overlap);

        // store results
        buffer->download();
        overlaps->download();

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t r=0; r<current_height; ++r)
        {
            unsigned int srcID = r+offset_h;
            L3DPP::pairwise_matches scored_matches;
            int new_matches = 0;

            for(size_t c=0; c<width; ++c)
            {
                // check depths -> must be bigger 0 (in front of cameras)
                float4 depths = buffer->dataCPU(c,r)[0];
                if(depths.x > 0.0f && depths.y > 0.0f && depths.z > 0.0f && depths.w > 0.0f)
                {
                    float overlap = overlaps->dataCPU(c,r)[0];

                    // potential match
                    L3DPP::Match M;
                    M.src_camID_ = srcCamID;
                    M.src_segID_ = srcID;
                    M.tgt_camID_ = tgtCamID;
                    M.tgt_segID_ = c;
                    M.overlap_score_ = overlap;
                    M.score3D_ = 0.0f;
                    M.depth_p1_ = depths.x;
                    M.depth_p2_ = depths.y;
                    M.depth_q1_ = depths.z;
                    M.depth_q2_ = depths.w;

                    if(kNN > 0)
                    {
                        // kNN matching
                        scored_matches.push(M);
                    }
                    else
                    {
                        // all matches are used
                        matches->at(srcID).push_back(M);
                        ++new_matches;
                    }
                }
            }

            // push kNN matches into list
            if(kNN > 0)
            {
                while(new_matches < kNN && !scored_matches.empty())
                {
                    matches->at(r).push_back(scored_matches.top());
                    scored_matches.pop();
                    ++new_matches;
                }
            }

            match_mutex.lock();
            num_matches += new_matches;
            match_mutex.unlock();
        }
    }

    // cleanup
    delete buffer;
    delete overlaps;

    return num_matches;
}

////////////////////////////////////////////////////////////////////////////////
void score_matches_GPU(L3DPP::DataArray<float4>* lines, L3DPP::DataArray<float4>* matches,
                       L3DPP::DataArray<int2>* ranges, L3DPP::DataArray<float>* scores,
                       L3DPP::DataArray<float2>* regularizers_tgt,
                       L3DPP::DataArray<float>* RtKinv, const float3 C,
                       const float two_sigA_sqr,
                       const float k, const float min_similarity)
{
    // init
    unsigned int block_size = L3D_BLOCK_SIZE;
    int width = matches->width();

    // define grid
    dim3 dimBlock = dim3(block_size*block_size,1);
    dim3 dimGrid = dim3(divUp(width, dimBlock.x),
                        divUp(1, dimBlock.y));

    // score matches
    L3DPP::K_score_matches <<< dimGrid, dimBlock >>> (width,lines->dataGPU(),
                                                      matches->dataGPU(),scores->dataGPU(),
                                                      ranges->dataGPU(),regularizers_tgt->dataGPU(),
                                                      RtKinv->dataGPU(),
                                                      RtKinv->strideGPU(),C,two_sigA_sqr,
                                                      k,min_similarity);
}

////////////////////////////////////////////////////////////////////////////////
void find_collinear_segments_GPU(L3DPP::DataArray<char>* C,
                                 L3DPP::DataArray<float4>* lines,
                                 const float dist_t)
{
    // init
    unsigned int block_size = L3D_BLOCK_SIZE;
    int size = lines->width();

    // define grid
    dim3 dimBlock = dim3(block_size,block_size);
    dim3 dimGrid = dim3(divUp(size, dimBlock.x),
                        divUp(size, dimBlock.y));

    // find collinear segments
    L3DPP::K_collinearity <<< dimGrid, dimBlock >>> (lines->dataGPU(),
                                                     C->dataGPU(),
                                                     C->strideGPU(),
                                                     size,dist_t);
}

////////////////////////////////////////////////////////////////////////////////
void replicator_dynamics_diffusion_GPU(L3DPP::SparseMatrix* &W, const std::string prefix)
{
    // init
    unsigned int block_size = L3D_BLOCK_SIZE;
    unsigned int num_rows_cols = W->num_rows_cols();
    unsigned int num_entries = W->num_entries();
    dim3 dimBlock = dim3(1,block_size*block_size);
    dim3 dimGrid_RC = dim3(divUp(1, dimBlock.x),
                           divUp(num_rows_cols, dimBlock.y));
    dim3 dimGrid = dim3(divUp(1, dimBlock.x),
                        divUp(num_entries, dimBlock.y));

    // create P matrix
    L3DPP::SparseMatrix* P = new L3DPP::SparseMatrix(W,true);

    // make copy of P
    L3DPP::SparseMatrix* P_prime = new L3DPP::SparseMatrix(P);

    // row normalize
    L3DPP::K_sparseMat_row_normalization <<< dimGrid_RC, dimBlock >>> (P->entries()->dataGPU(),
                                                                       P->start_indices()->dataGPU(),
                                                                       num_rows_cols,num_entries);

    cudaDeviceSynchronize();

    for(int i=0; i<L3D_DEF_RDD_MAX_ITER; ++i)
    {
        // diffusion
        std::cout << prefix << "iteration: " << i << std::endl;

        // update
        L3DPP::K_sparseMat_diffusion_step <<< dimGrid, dimBlock >>> (P->entries()->dataGPU(),W->entries()->dataGPU(),
                                                                     P->start_indices()->dataGPU(),W->start_indices()->dataGPU(),
                                                                     P_prime->entries()->dataGPU(),P_prime->start_indices()->dataGPU(),
                                                                     num_entries);

        cudaDeviceSynchronize();

        // row normalize
        L3DPP::SparseMatrix* tmp = P;
        P = P_prime;
        P_prime = tmp;

        if(i < L3D_DEF_RDD_MAX_ITER-1)
        {
            L3DPP::K_sparseMat_row_normalization <<< dimGrid_RC, dimBlock >>> (P->entries()->dataGPU(),
                                                                               P->start_indices()->dataGPU(),
                                                                               num_rows_cols,num_entries);
        }

        cudaDeviceSynchronize();
    }

    // re-assign
    delete W;
    W = P;

    delete P_prime;
}

}

#endif //L3DPP_CUDA
