#include <stdio.h>
#include "integrals.h"

template <class Tin, class Tout, unsigned int blockSize>
__global__ void reduceCKernel(Tin *g_idata, Tout *g_odata, int xs, int ys, unsigned int xside, unsigned int grid_dimx)
{
    Tout *sdata = SharedMemory<Tout>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
   // unsigned int tid = threadIdx.x;
    //unsigned int gridSize = blockSize*2*gridDim.x;

    unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

#define cellSizex (blockSize*2)
	unsigned int ix = xs + blockIdx.x*cellSizex + tx;
	unsigned int iy = ys + blockIdx.y*blockSize + ty;

	unsigned int ind = ix + xside*iy;
	unsigned int tind = tx + blockSize*ty;


    Tout mySum(g_idata, ind);
	mySum.Add(g_idata, ind+blockSize);

    // each thread puts its local sum into shared memory 
    sdata[tind] = mySum;
    __syncthreads();


    volatile Tout* smem = sdata;
	if(tx < 16)
	{
		if (blockSize >=  32) { smem[tind] = mySum = mySum + smem[tind + 16]; }
		if (blockSize >=  16) { smem[tind] = mySum = mySum + smem[tind +  8]; }
		if (blockSize >=   8) { smem[tind] = mySum = mySum + smem[tind +  4]; }
		if (blockSize >=   4) { smem[tind] = mySum = mySum + smem[tind +  2]; }
		if (blockSize >=   2) { smem[tind] = mySum = mySum + smem[tind +  1]; }
	}
	__syncthreads();

	unsigned int tiy = blockSize*ty;

	if(ty < 16)
	{
		if (blockSize >=  32) { smem[tiy] = mySum = mySum + smem[tiy + 16*blockSize];  __syncthreads();}
		if (blockSize >=  16) { smem[tiy] = mySum = mySum + smem[tiy +  8*blockSize];  __syncthreads();}
		if (blockSize >=   8) { smem[tiy] = mySum = mySum + smem[tiy +  4*blockSize];  __syncthreads();}
		if (blockSize >=   4) { smem[tiy] = mySum = mySum + smem[tiy +  2*blockSize];  __syncthreads();}
		if (blockSize >=   2) { smem[tiy] = mySum = mySum + smem[tiy +  1*blockSize];  __syncthreads();}
	}

	if(tind == 0)
		g_odata[blockIdx.x + blockIdx.y*grid_dimx] = sdata[0];
}


extern "C"
bool isPow2(unsigned int x);


//use 256, 64 threads/blocks,
template <class Tin, class Tout>
void reduceSmallC(int xside, int yside, int blockSize, Tin *d_idata, Tout *d_odata)
{
	int blocksx = xside/cellSizex;
	int blocksy = yside/blockSize;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(blocksx, blocksy, 1);

	
	int smemSize = blockSize*blockSize* sizeof(Tout);
	cudaFuncSetCacheConfig(reduceCKernel<Tin, Tout,  32>, cudaFuncCachePreferShared );

    reduceCKernel<Tin, Tout,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, 0, 0, xside, blocksx);
}

template void reduceSmallC<float4, TCMax<0>>(int xside, int yside, int blockSize, float4 *d_idata, TCMax<0> *d_odata);

extern float *tempHostData;

//block size is 32
template <class Tin, class Tout>
__global__ void reduce32(int xs, int ys, Tin *g_idata, Tout *g_odata, int xside)
{
    Tout *sdata = SharedMemory<Tout>();

    unsigned int tid = threadIdx.x;
	int ix = xs + blockIdx.x*32*2 + threadIdx.x;
	int iy = ys + blockIdx.y;
    unsigned int i = ix + iy*xside; 
 
	Tout mySum(g_idata, i);
	mySum.Add(g_idata, i+32);  

    volatile Tout* smem = sdata;

    smem[tid] = mySum;

    smem[tid] = mySum = mySum + smem[tid + 16];
    smem[tid] = mySum = mySum + smem[tid +  8];
    smem[tid] = mySum = mySum + smem[tid +  4];
    smem[tid] = mySum = mySum + smem[tid +  2];
    smem[tid] = mySum = mySum + smem[tid +  1];
    
    if (tid == 0) 
        g_odata[blockIdx.x + blockIdx.y*gridDim.x] = sdata[0];
}

template <class Tin, class Tout> void Reduce2DBlock(int xs, int ys, int widthx, int widthy,
													int xside, int& nNewBlocks,
													Tin* cuInputFiels, Tout* cuTempData, Tout* hostOutData)
{
	int blockSize = 32;
	int blocksx = widthx/cellSizex;
	int blocksy = widthy; 

    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(blocksx, blocksy, 1);
	int smemSize = 2*blockSize*sizeof(LSum);

	reduce32<Tin, Tout><<< dimGrid, dimBlock, smemSize >>>(xs, ys, cuInputFiels, (Tout*)cuTempData, xside);
//-----
	int nBlocks = blocksx*blocksy;
	nNewBlocks = nBlocks/cellSizex;

    dimGrid = dim3(nNewBlocks, 1, 1);
	Tout* cuNewTemp = ((Tout*)cuTempData) + nBlocks;

	reduce32<Tout, Tout><<< dimGrid, dimBlock, smemSize >>>(0, 0, cuTempData, cuNewTemp, nBlocks);

	cudaMemcpy(hostOutData, cuNewTemp, nNewBlocks*sizeof(LSum), cudaMemcpyDeviceToHost);
}


extern float *tempHostData;

int localSum(int xc, int yc, int width, int xside, int yside, float4 *field, LSum* cuTempData, float* fsum)
{
	int xs = max(0, xc - width/2);
	int ys = max(0, yc - width/2);

	int xe = min(xc + width/2, xside-1);
	int ye = min(yc + width/2, yside-1);

	width = xe-xs;
	width -= width%64;

	memset(fsum, 0, 8*sizeof(float));

	if(width==0)
		return -1;

	xs = xc - width/2;
	ys = yc - width/2;
	int nNewBlocks;

	if(xs < 0 || xs+width >= xside)
		return -1;

	if(ys < 0 || ys+width >= yside)
		return -1;

	LSum* hdata = (LSum*)tempHostData;
	Reduce2DBlock<float4, LSum>(xs, ys, width, width,
								xside, nNewBlocks,
								field, cuTempData, hdata);


	for(int i = 0; i < nNewBlocks; i++)
	for(int fi = 0; fi < 8; fi++)
	{
		fsum[fi] += hdata[i].v[fi];
	}

	return 0;
}


#include "layer_defines.h"
void Reduce2DBlockH(int xs, int ys, int widthx, int widthy,
				   int xside, int& nNewBlocks,
				   float4* cuInputFiels, float* cuTempData, TCMaxHost* hostOutData)
{

	Reduce2DBlock<float4, TCMax<0>>(xs, ys, widthx, widthy,
				   xside, nNewBlocks,
				   cuInputFiels, (TCMax<0>*)cuTempData, (TCMax<0>*)hostOutData);
};

void Reduce2DBlockYG(int xs, int ys, int widthx, int widthy,
				   int xside, int& nNewBlocks,
				   float4* cuInputFiels, float* cuTempData, TCMaxHost* hostOutData)
{

	Reduce2DBlock<float4, TCMax<1>>(xs, ys, widthx, widthy,
				   xside, nNewBlocks,
				   cuInputFiels, (TCMax<1>*)cuTempData, (TCMax<1>*)hostOutData);
};

