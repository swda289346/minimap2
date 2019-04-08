#ifndef CHAIN_CUDA_KERNEL_H
#define CHAIN_CUDA_KERNEL_H

__global__ void mm_chain_dp_fill_cuda_score(int64_t i, uint64_t ri, int32_t qi, int32_t q_span, int32_t sidi, int64_t *mem_max_j, int32_t *mem_max_f, int max_dist_x, int max_dist_y, int bw, int max_iter, int n_segs, int is_cdna, mm128_t *a, float avg_qspan, int32_t *f, int32_t *t, int32_t *p);
__global__ void mm_chain_dp_fill_cuda_max(int i, int64_t *mem_max_j, int32_t *mem_max_f, int max_iter, int32_t *f, int32_t *p, int32_t *v);

#endif
