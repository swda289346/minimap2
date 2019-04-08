#include <cuda.h>
#include "minimap.h"
#include "mmpriv.h"
#include "kalloc.h"
#include "chain_cuda_kernel.h"

extern "C" {

void mm_chain_dp_fill_cuda(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int n_segs, int is_cdna, int64_t n, mm128_t *a, float avg_qspan, int32_t *f, int32_t *t, int32_t *p, int32_t *v)
{
	const int blockDim = 32;
	int numBlocks = (max_iter+blockDim-1)/blockDim;
	int64_t j;
	int32_t *f_cuda, *t_cuda, *p_cuda, *v_cuda;
	mm128_t *a_cuda;
	cudaError_t result;
	for (uint64_t i=0;i<n;i++)
	{
		v[i] = f[i] = a[i].y>>32&0xff; // NB: only 8 bits of span is used!!!
		p[i] = -1;
	}
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
	result = cudaDeviceSynchronize();
	assert(result==cudaSuccess);
	for (j = 0; j < n-1; ++j) {
		mm_chain_dp_fill_cuda_optimized<<<numBlocks, blockDim>>>(j, n, max_dist_x, max_dist_y, bw, max_iter, n_segs, is_cdna, a_cuda, avg_qspan, f_cuda, t_cuda, p_cuda, v_cuda);
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
}

}

