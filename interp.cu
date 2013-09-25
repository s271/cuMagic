#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "mmath.cuh"
#include "layer_defines.h"

__device__ inline float FM1(const float x)
{
	return -x*x*x + 2*x*x - x;
}

__device__ inline float F0(const float x)
{
	return 3*x*x*x - 5*x*x + 2;
}

__device__ inline float FP1(const float x)
{
	return -3*x*x*x + 4*x*x + x;
}

__device__ inline float FP2(const float x)
{
	return x*x*x - x*x;
}

__device__ inline float spline(const float x, const float pm, const float p0, const float p1, const float p2)
{
	return .5f*(pm*FM1(x) + p0*F0(x) + p1*FP1(x) + p2*FP2(x));
}

__global__ void StageTKernel(float* inp, float* out, float invscalein, float scaleout, int intsizex, int intsizey, int ysideout)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ixint = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ixint >= intsizex)
		return;

	if(iy >= ysideout)
		return;
 

	float fy = invscalein*iy*scaleout;

	int iyint = fy;

	fy -= iyint;

	int iyintm = (iyint-1 + intsizey)%intsizey;
	int iyintp = (iyint + 1)%intsizey; 
	int iyintp2 = (iyint + 2)%intsizey; 

	float val = spline(fy, inp[ixint + iyintm*intsizex],
		inp[ixint + iyint*intsizex],
		inp[ixint + iyintp*intsizex],
		inp[ixint + iyintp2*intsizex]);

	int ind = ixint + intsizex*iy;
	out[ind] =  val;
}

__global__ void StageUKernel(float* inp, float* out, float invscalein, float scaleout, float scaleadd, 
							 int intsizex, int intsizey, int xsideout, int ysideout)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ix >= xsideout)
		return;

	if(iy >= ysideout)
		return;
 

	float fx = invscalein*ix*scaleout;

	int ixint = fx;
	fx -= ixint;


	int ixintm = (ixint-1 + intsizex)%intsizex;
	int ixintp = (ixint + 1)%intsizex; 
	int ixintp2 = (ixint + 2)%intsizex; 

	float val = spline(fx, inp[ixintm + iy*intsizex],
		inp[ixint + iy*intsizex],
		inp[ixintp + iy*intsizex],
		inp[ixintp2 + iy*intsizex]);

	int ind = ix + xsideout*iy;
	out[ind] += scaleadd*val;

	out[ind] = fminf(fmaxf((out[ind]-.5)*1.1+.5,0), 1);
}

void Spline2D(float* interpGrid, int intsizex, int intsizey, float* temp_grid, float scaleadd, float* out, int xsideout, int ysideout)
{
	int blockx = 16;
	int blocky = 16;

    dim3 gridt(intsizex/blockx + 1, ysideout/blocky + 1
		);
    dim3 blockt(blockx, blocky);

	float scaleout = 1.f;
	float invscalein = intsizey*1.f/ysideout;

	StageTKernel<<< gridt, blockt >>>(interpGrid, temp_grid, invscalein, scaleout, intsizex, intsizey, ysideout);

	scaleout = 1.f;
	invscalein = intsizex*1.f/xsideout;

    dim3 gridu(xsideout/blockx + 1, ysideout/blocky + 1);
    dim3 blocku(blockx, blocky);

	StageUKernel<<< gridu, blocku >>>(temp_grid, out, invscalein, scaleout, scaleadd, intsizex, intsizey, xsideout, ysideout);

};
