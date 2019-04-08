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
	int real_iter = numBlocks*blockDim;
	const int n_gpu_max = 1<<24;
	int n_gpu = n<n_gpu_max?n:n_gpu_max;
	int64_t j;
	int64_t start;
	int32_t *f_cuda, *t_cuda, *p_cuda, *v_cuda;
	mm128_t *a_cuda;
	cudaError_t result;
	for (uint64_t i=0;i<n;i++)
	{
		v[i] = f[i] = a[i].y>>32&0xff; // NB: only 8 bits of span is used!!!
		p[i] = -1;
	}
	result = cudaMalloc(&f_cuda, sizeof(int32_t)*n_gpu);
	assert(result==cudaSuccess);
	result = cudaMalloc(&t_cuda, sizeof(int32_t)*n_gpu);
	assert(result==cudaSuccess);
	result = cudaMalloc(&p_cuda, sizeof(int32_t)*n_gpu);
	assert(result==cudaSuccess);
	result = cudaMalloc(&v_cuda, sizeof(int32_t)*n_gpu);
	assert(result==cudaSuccess);
	result = cudaMalloc(&a_cuda, sizeof(mm128_t)*n_gpu);
	assert(result==cudaSuccess);
	for (start=0;start<n;)
	{
		if (start && start+real_iter>=n)
			break;
		int n_process = n-start;
		int n_copy = n_process;
		if (n_process>n_gpu)
		{
			n_process = n_gpu-real_iter;
			n_copy = n_gpu;
		}
		result = cudaMemcpy(f_cuda, f+start, sizeof(int32_t)*n_copy, cudaMemcpyHostToDevice);
		assert(result==cudaSuccess);
		result = cudaMemcpy(t_cuda, t+start, sizeof(int32_t)*n_copy, cudaMemcpyHostToDevice);
		assert(result==cudaSuccess);
		result = cudaMemcpy(p_cuda, p+start, sizeof(int32_t)*n_copy, cudaMemcpyHostToDevice);
		assert(result==cudaSuccess);
		result = cudaMemcpy(v_cuda, v+start, sizeof(int32_t)*n_copy, cudaMemcpyHostToDevice);
		assert(result==cudaSuccess);
		result = cudaMemcpy(a_cuda, a+start, sizeof(mm128_t)*n_copy, cudaMemcpyHostToDevice);
		assert(result==cudaSuccess);
		cudaDeviceSynchronize();
		assert(result==cudaSuccess);
		for (j = start==0?0:start+real_iter; j < start+n_process; ++j) {
			mm_chain_dp_fill_cuda_optimized_batch<<<numBlocks, blockDim>>>(j, n, start, n_process, max_dist_x, max_dist_y, bw, max_iter, n_segs, is_cdna, a_cuda, avg_qspan, f_cuda, t_cuda, p_cuda, v_cuda);
		}
		cudaDeviceSynchronize();
		assert(result==cudaSuccess);
		result = cudaMemcpy(f+start, f_cuda, sizeof(int32_t)*n_copy, cudaMemcpyDeviceToHost);
		assert(result==cudaSuccess);
		result = cudaMemcpy(t+start, t_cuda, sizeof(int32_t)*n_copy, cudaMemcpyDeviceToHost);
		assert(result==cudaSuccess);
		result = cudaMemcpy(p+start, p_cuda, sizeof(int32_t)*n_copy, cudaMemcpyDeviceToHost);
		assert(result==cudaSuccess);
		result = cudaMemcpy(v+start, v_cuda, sizeof(int32_t)*n_copy, cudaMemcpyDeviceToHost);
		assert(result==cudaSuccess);
		start += n_process-real_iter;
	}
	cudaFree(f_cuda);
	cudaFree(t_cuda);
	cudaFree(p_cuda);
	cudaFree(v_cuda);
	cudaFree(a_cuda);
}

}

