#include <cuda.h>
#include "minimap.h"
#include "mmpriv.h"
#include "kalloc.h"

__global__ void mm_chain_dp_fill_cuda_score(int64_t i, uint64_t ri, int32_t qi, int32_t q_span, int32_t sidi, int64_t *mem_max_j, int32_t *mem_max_f, int max_dist_x, int max_dist_y, int bw, int max_iter, int n_segs, int is_cdna, mm128_t *a, float avg_qspan, int32_t *f, int32_t *t, int32_t *p)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if (tid>=max_iter)
		return;
	int j = i-1-tid;
	mem_max_j[tid] = -1;
	mem_max_f[tid] = q_span;
	if (j<0)
		return;
	if (ri > a[j].x + max_dist_x)
		return;
	int32_t min_d;
	int64_t dr = ri - a[j].x;
	int32_t dq = qi - (int32_t)a[j].y, dd, sc, log_dd;
	int32_t sidj = (a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
	if ((sidi == sidj && dr == 0) || dq <= 0) return; // don't skip if an anchor is used by multiple segments; see below
	if ((sidi == sidj && dq > max_dist_y) || dq > max_dist_x) return;
	dd = dr > dq? dr - dq : dq - dr;
	if (sidi == sidj && dd > bw) return;
	if (n_segs > 1 && !is_cdna && sidi == sidj && dr > max_dist_y) return;
	min_d = dq < dr? dq : dr;
	sc = min_d > q_span? q_span : dq < dr? dq : dr;
	log_dd = dd? 31-__clz(dd) : 0;
	if (is_cdna || sidi != sidj) {
		int c_log, c_lin;
		c_lin = (int)(dd * .01 * avg_qspan);
		c_log = log_dd;
		if (sidi != sidj && dr == 0) ++sc; // possibly due to overlapping paired ends; give a minor bonus
		else if (dr > dq || sidi != sidj) sc -= c_lin < c_log? c_lin : c_log;
		else sc -= c_lin + (c_log>>1);
	} else sc -= (int)(dd * .01 * avg_qspan) + (log_dd>>1);
	sc += f[j];
	if (sc > mem_max_f[tid]) {
		mem_max_f[tid] = sc, mem_max_j[tid] = j;
	}
	if (p[j] >= 0) t[p[j]] = i;
}

__global__ void mm_chain_dp_fill_cuda_max(int i, int64_t *mem_max_j, int32_t *mem_max_f, int max_iter, int32_t *f, int32_t *p, int32_t *v)
{
	__shared__ int64_t max_j[1024];
	__shared__ int32_t max_f[1024];
	int tid = threadIdx.x;
	int stride = blockDim.x;
	if (tid<max_iter)
	{
		max_j[tid] = mem_max_j[tid];
		max_f[tid] = mem_max_f[tid];
	}
	else
	{
		max_j[tid] = -1;
		max_f[tid] = 0;
	}
	for (int k=stride+tid;k<max_iter;k+=stride)
		if (mem_max_f[k]>max_f[tid])
		{
			max_j[tid] = mem_max_j[k];
			max_f[tid] = mem_max_f[k];
		}
	for (unsigned int offset=stride>>1;offset>0;offset>>=1)
	{
		__syncthreads();
		if (tid<offset)
			if (max_f[tid+offset]>max_f[tid]||(max_f[tid+offset]==max_f[tid]&&max_j[tid+offset]>max_j[tid]))
			{
				max_j[tid] = max_j[tid+offset];
				max_f[tid] = max_f[tid+offset];
			}
	}
	__syncthreads();
	if (tid==0)
	{
		f[i] = max_f[0], p[i] = max_j[0];
		v[i] = max_j[0] >= 0 && v[max_j[0]] > max_f[0]? v[max_j[0]] : max_f[0]; // v[] keeps the peak score up to i; f[] is the score ending at i, not always the peak
	}
}

