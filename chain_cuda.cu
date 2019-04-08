#include <cuda.h>
#include "minimap.h"
#include "mmpriv.h"
#include "kalloc.h"
#include "chain_cuda_kernel.h"

extern "C" {

void mm_chain_dp_fill_cuda(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int n_segs, int is_cdna, int64_t n, mm128_t *a, float avg_qspan, int32_t *f, int32_t *t, int32_t *p, int32_t *v)
{
	const int blockDim = 32;
	const int maxBlockDim = 1024;
	int numBlocks = (max_iter+blockDim-1)/blockDim;
	int64_t i;
	int32_t *f_cuda, *t_cuda, *p_cuda, *v_cuda;
	mm128_t *a_cuda;
	int64_t *mem_max_j;
	int32_t *mem_max_f;
	cudaError_t result;
	result = cudaMalloc(&f_cuda, sizeof(int32_t)*n);
	assert(result==cudaSuccess);
	result = cudaMalloc(&t_cuda, sizeof(int32_t)*n);
	assert(result==cudaSuccess);
	result = cudaMalloc(&p_cuda, sizeof(int32_t)*n);
	assert(result==cudaSuccess);
	result = cudaMalloc(&v_cuda, sizeof(int32_t)*n);
	assert(result==cudaSuccess);
	result = cudaMalloc(&a_cuda, sizeof(mm128_t)*n);
	assert(result==cudaSuccess);
	result = cudaMemcpy(f_cuda, f, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
	assert(result==cudaSuccess);
	result = cudaMemcpy(t_cuda, t, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
	assert(result==cudaSuccess);
	result = cudaMemcpy(p_cuda, p, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
	assert(result==cudaSuccess);
	result = cudaMemcpy(v_cuda, v, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
	assert(result==cudaSuccess);
	result = cudaMemcpy(a_cuda, a, sizeof(mm128_t)*n, cudaMemcpyHostToDevice);
	assert(result==cudaSuccess);
	result = cudaMalloc(&mem_max_j, sizeof(int64_t)*max_iter);
	assert(result==cudaSuccess);
	result = cudaMalloc(&mem_max_f, sizeof(int32_t)*max_iter);
	assert(result==cudaSuccess);
	result = cudaDeviceSynchronize();
	assert(result==cudaSuccess);
	for (i = 0; i < n; ++i) {
		uint64_t ri = a[i].x;
		int32_t qi = (int32_t)a[i].y, q_span = a[i].y>>32&0xff; // NB: only 8 bits of span is used!!!
		int32_t sidi = (a[i].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
		mm_chain_dp_fill_cuda_score<<<numBlocks, blockDim>>>(i, ri, qi, q_span, sidi, mem_max_j, mem_max_f, max_dist_x, max_dist_y, bw, max_iter, n_segs, is_cdna, a_cuda, avg_qspan, f_cuda, t_cuda, p_cuda);
		mm_chain_dp_fill_cuda_max<<<1, maxBlockDim>>>(i, mem_max_j, mem_max_f, max_iter, f_cuda, p_cuda, v_cuda);
	}
	result = cudaDeviceSynchronize();
	assert(result==cudaSuccess);
	result = cudaMemcpy(f, f_cuda, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
	assert(result==cudaSuccess);
	result = cudaMemcpy(t, t_cuda, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
	assert(result==cudaSuccess);
	result = cudaMemcpy(p, p_cuda, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
	assert(result==cudaSuccess);
	result = cudaMemcpy(v, v_cuda, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
	assert(result==cudaSuccess);
	cudaFree(f_cuda);
	cudaFree(t_cuda);
	cudaFree(p_cuda);
	cudaFree(v_cuda);
	cudaFree(a_cuda);
	cudaFree(mem_max_j);
	cudaFree(mem_max_f);
}

}

