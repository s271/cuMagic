/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/


#include <stdio.h>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type

#include "integrals.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/


/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class Tin, class Tout, unsigned int blockSize, bool nIsPow2, bool isCoordPres>
__global__ void reduce6(Tin *g_idata, Tout *g_odata, unsigned int n)
{
    Tout *sdata = SharedMemory<Tout>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    //Tout mySum = 0;
	Tout mySum(0);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
	if(!isCoordPres)
	{
		while (i < n)
		{         
			mySum += g_idata[i];
			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n) 
				mySum += g_idata[i+blockSize];  
			i += gridSize;
		} 
	}
	else
	{
		while (i < n)
		{         
			mySum.Add(g_idata, i);
			//// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n) 
				mySum.Add(g_idata, i+blockSize);  
			i += gridSize;
		}	
	}

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile Tout* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

//use 256, 64 threads/blocks,
template <class Tin, class Tout, bool isCoord>
void reduce(int size, int threads, int blocks, Tin *d_idata, Tout *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(Tout) : threads * sizeof(Tout);

    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            reduce6<Tin, Tout, 512, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 256:
            reduce6<Tin, Tout, 256, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 128:
            reduce6<Tin, Tout, 128, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 64:
            reduce6<Tin, Tout,  64, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 32:
            reduce6<Tin, Tout,  32, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 16:
            reduce6<Tin, Tout,  16, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  8:
            reduce6<Tin, Tout,   8, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  4:
            reduce6<Tin, Tout,   4, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  2:
            reduce6<Tin, Tout,   2, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  1:
            reduce6<Tin, Tout,   1, true, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            reduce6<Tin, Tout, 512, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 256:
            reduce6<Tin, Tout, 256, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 128:
            reduce6<Tin, Tout, 128, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 64:
            reduce6<Tin, Tout,  64, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 32:
            reduce6<Tin, Tout,  32, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 16:
            reduce6<Tin, Tout,  16, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  8:
            reduce6<Tin, Tout,   8, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  4:
            reduce6<Tin, Tout,   4, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  2:
            reduce6<Tin, Tout,   2, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  1:
            reduce6<Tin, Tout,   1, false, isCoord><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        }
    }
}


bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

template void 
reduce<float4, LSum, false>(int size, int threads, int blocks, 
           float4 *d_idata, LSum *d_odata);
              
template void 
reduce<float4, TCMax<0>, true>(int size, int threads, int blocks, 
           float4 *d_idata, TCMax<0> *d_odata);