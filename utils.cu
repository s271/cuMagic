#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "mmath.cuh"
#include "layer_defines.h"

__global__ void RandFKernel(unsigned int seed, float* temp, int xside, int yside)
{
    int tx = threadIdx.x;
	int bw = blockDim.x;
	int ix = blockIdx.x*bw + tx;
	if(ix >= xside)
		return;

	unsigned int rng_state =380116161 + seed + 11137*ix + 117*seed*ix*(ix+seed);

	for(int iy = 0; iy < yside; iy++)
	{
		rand_xorshift(rng_state);
		// Generate a random float in [0, 1)...
		float f0 = float(rng_state) * I2F;
		int ind = ix + xside*iy;
		temp[ind] = f0;
	}
}

void InitRnd2DF(int seed, float* temp, int intsizex, int intsizey)
{
	int blockx = 128;
	dim3 grid(intsizex/blockx + (intsizex%blockx)?1:0);
    dim3 block(blockx);

	RandFKernel<<< grid, block >>>(seed, temp, intsizex, intsizey);

};


__global__ void RandIntKernel(unsigned int seed, unsigned int* temp, int xside, int yside)
{
    int tx = threadIdx.x;
	int bw = blockDim.x;
	int ix = blockIdx.x*bw + tx;

	unsigned int rng_state =113 + seed + 11137117*ix + 117*seed*ix*(ix+seed);

	for(int iy = 0; iy < yside; iy++)
	{
		rand_xorshift(rng_state);
		int ind = ix + xside*iy;
		temp[ind] = rng_state;
	}
}

void InitRnd2DInt(int seed, unsigned int* temp, int intsizex, int intsizey)
{
	int blockx = 256;
	dim3 grid(intsizex/blockx);
    dim3 block(blockx);

	RandIntKernel<<< grid, block >>>(seed, temp, intsizex, intsizey);

};

__global__ void RandomizeKernel(unsigned int* cuRand, int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ix >= xside || iy >= yside)
		return;

	int ind = ix + xside*iy;

	unsigned int rnd = cuRand[ind];
	rand_xorshift(rnd);
	rand_xorshift(rnd);
	cuRand[ind] = rnd;

}


void Randomize(unsigned int* cuRand, int imgw, int imgh)
{
	int blockx = 64;
	int blocky = 8;

    dim3 grid((imgw/blockx)+(!(imgw%blockx)?0:1), (imgh/blocky)+(!(imgh%blocky)?0:1));
    dim3 block(blockx, blocky);

	RandomizeKernel<<< grid, block >>> (cuRand, imgw, imgh);

};

//-----------------
__global__ void TempWavePackKernel(int nf, float attf, float* cuTemp, float sc)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int ix = blockIdx.x*bw + tx;
	float x = ix*sc;
	float compx =sqrtf(x);
	cuTemp[ix] = expf(-x*.5f)*cosf(nf*compx*M_PI);
}


void InitWavePack(int nf, float attf, int sim_width, int sim_height, float* temp, cudaArray* out)
{
	int blockx = 64;

    dim3 grid((sim_width/blockx)+(!(sim_width%blockx)?0:1));
    dim3 block(blockx);
	float sc = 1.f/sim_height;
	TempWavePackKernel<<< grid, block >>> (nf, attf, temp, sc);

	cudaMemcpyToArray(out, 0, 0, temp, sizeof(float)*sim_width, cudaMemcpyDeviceToDevice);
};
//-----------------

__global__ void SmoothKernel(float attf, float* cuTemp, float inv_w, int off)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int ix = blockIdx.x*bw + tx;
	float x = (ix-off)*inv_w*4*attf;
	cuTemp[ix] = atan(x);
}

void InitSmooth(float attf, int sim_width, float* temp, cudaArray* out)
{
	int blockx = 64;

    dim3 grid((sim_width/blockx)+(!(sim_width%blockx)?0:1));
    dim3 block(blockx);
	float sc = 1.f/sim_height;
	SmoothKernel<<< grid, block >>> (attf, temp, 1./sim_width, sim_width/2);
	cudaMemcpyToArray(out, 0, 0, temp, sizeof(float)*sim_width, cudaMemcpyDeviceToDevice);

};

//-----------------

__global__ void InitSpiral2DKernel(float* cuRes, float cx, float cy,
								   float omega, float inv_branches, float r2,float rmax,
								   int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;
	float fx = ix - cx;
	float fy = iy - cy;
	float d2 = fx*fx + fy*fy;
	if(d2 > r2)
		return;

	int ind = ix + xside*iy;

	float r = sqrtf(d2);
	float phi = atan2(fy, fx);
	float t = phi+omega*r;
	cuRes[ind] = fminf(fmaxf(1.1f*cosf(inv_branches*t),-1),1)*.5+.5;
}


void InitFuncLayer(cudaArray* cuFuncLayer, float* cuTemp, int xside, int yside)
{

	int blockx = 32;
	int blocky = 8;


    dim3 grid(sim_width/blockx, sim_height/blocky);
    dim3 block(blockx, blocky);     

	float cx = xside*.5f;
	float cy = yside*.5f;
	float omega = 6*M_PI/xside;
	int nbranches  = 6;
	float rad = xside*.1f;
	float r2 = rad*rad;

	cudaMemset(cuTemp, 0, sim_rect*sizeof(float));

    InitSpiral2DKernel<<< grid, block >>> (cuTemp, cx, cy, omega, nbranches, r2, rad, xside, yside);


	cudaMemcpyToArray(cuFuncLayer, 0, 0, cuTemp, sizeof(float)*sim_rect, cudaMemcpyDeviceToDevice);
};

