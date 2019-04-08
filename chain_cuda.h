#ifndef CHAIN_CUDA_H
#define CHAIN_CUDA_H

void mm_chain_dp_fill_cuda(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int n_segs, int is_cdna, int64_t n, mm128_t *a, float avg_qspan, int32_t *f, int32_t *t, int32_t *p, int32_t *v);

#endif

