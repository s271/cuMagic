#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "mmath.cuh"
#include "layer_defines.h"
#include "glcontrols.h"

#define FILL_SHARE .2f;

texture<float, 1, cudaReadModeElementType> smoothFunc;
texture<float4, 2, cudaReadModeElementType> vectTex0;

#if NFLAYERS ==2
	texture<float2, 2, cudaReadModeElementType> flTex;
#else if NFLAYERS ==4
	texture<float4, 2, cudaReadModeElementType> flTex;
#endif

__device__ int interactMatrixF[PMAX][PMAX];
__device__ int interactMatrixP[PMAX][PMAX];


__global__ void DistribureParticlesKernel(unsigned int seed, int warp, float* cuPhysLayer,
										  unsigned int* cuStateLayer, unsigned int* cuRand, int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	int ind = ix + xside*iy;
	unsigned int rnd_seed = seed + cuRand[ind];
	rand_xorshift(rnd_seed);
	if(rnd_seed%warp == 0)
	{
		rand_xorshift(rnd_seed);
		float val = MAX_PL_VALUE*(.3f + .4f*I2F*rnd_seed);
		cuPhysLayer[ind] = val;
	}
}

void DistribureParticles(float* cuPhysLayer, unsigned int* cuStateLayer, unsigned int* cuRand, int xside, int yside)
{
	int blockx = 32;
	int blocky = 8;

	unsigned int seed = rand();

    dim3 grid(sim_width/blockx, sim_height/blocky);
    dim3 block(blockx, blocky);     
	int warp = 1.f/FILL_SHARE;
    DistribureParticlesKernel<<< grid, block >>> (seed, warp, cuPhysLayer, cuStateLayer, cuRand, xside, yside);

};

void InitPhysLayer(float* cuPhysLayer, unsigned int* cuStateLayer, unsigned int* cuRand, int xside, int yside)
{
	cudaMemset(cuPhysLayer, 0, sim_rect*sizeof(float));
	cudaMemset(cuStateLayer, 0, sim_rect*sizeof(float));
	DistribureParticles(cuPhysLayer, cuStateLayer, cuRand, xside, yside);
};


#define RSH 4
#define RSH_CELL (2*RSH+1)
#define SMEM(X, Y) sdata[(Y)*tilew+(X)]

struct PValRnd
{
	float val;
	unsigned int signs;
};

#define DRES_INV .02f

#define SCOPY(xd, yd, xs, ys)\
	{\
	PValRnd& ref = SMEM(xd, yd);\
	int wind =  WARPX(xs) + xside*WARPY(ys);\
	ref.val = cuPhysLayerIn[wind];\
	ref.signs = cuState[wind];\
	}

__global__ void ParticleKernel(const float* cuPhysLayerIn,
							   const unsigned int* cuState, 
							   unsigned int* cuStateOut, 
							   float* __restrict__ fieldLayer,
							   float* __restrict__ vectField,
							   float* __restrict__ cuPhysLayerOut,
							   const float* cuLandscape, const unsigned int* cuRand, int xside, int yside)
{
	extern __shared__ PValRnd sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = DELTA_M + blockIdx.x*bw + tx;
    int iy = DELTA_M + blockIdx.y*bh + ty;
	int tilew = bw + 2*RSH;

	COPY_SHARED_MEM(ix, iy)
	__syncthreads();


	PValRnd& rc = SMEM(tx+RSH, ty+RSH);
	float inval_min = fmaxf(rc.val - MAX_PL_VALUE, 0);

	float outval = 0;
	unsigned int signsc = rc.signs;
	unsigned int color = signsc&15;

	float sumv = 0;
	float sum_yg = 0;
	float sum_ov = 0;
	float sum_mj = 0;

	for(int ox = 0; ox < RSH_CELL; ox++)
	for(int oy = 0; oy < RSH_CELL; oy++)
	{
		if(ox == RSH && oy ==RSH)
			continue;

		PValRnd& ref = SMEM(tx+ox, ty+oy);
		float val = fminf(ref.val, MAX_PL_VALUE);

		if(val==0)
			continue;

		unsigned int signs = ref.signs;
		int pcolor = signs&15;

		if(pcolor <= P_BLUE)
			sumv += val*(2*pcolor-3);//optimize
		else if(pcolor <= P_YELLOW )
			sum_yg -= val*(2*pcolor-7);
		else if(pcolor <= P_VIOLET)
			sum_ov -= val*(2*pcolor-11);
		else 
			sum_mj -= val*(2*pcolor-15);

		unsigned int dxi = (signs>>4)&15;
		unsigned int dyi = (signs>>8)&15;

		if(dxi == ox && dyi == oy)
		{
			if(val > outval)
				color = pcolor;

			outval += val;
		}
	}

	int ind = ix + xside*iy;
	unsigned int rnd = cuRand[ind];

	float vx = vectField[4*ind];
	float vy = vectField[4*ind+1];
	float vval = vectField[4*ind+2];
	float vmj = vectField[4*ind+3];

	if(vmj >= 0)
	{
		if(sum_yg !=0)
			vval = (1-2*signbit(sum_yg))*fabsf(vval);
	}
	else
	{
		if(sum_mj !=0)
			vval = (1-2*signbit(sum_mj))*fabsf(vval);
	}

	float fld_ov = fieldLayer[NFLAYERS*ind+2] + .6*sum_ov;

	if(outval <= MAX_PL_VALUE)
	{
		int rndpass = rnd & (1<<(32-9));
		if(fld_ov > .008  && rndpass)
		{
			color = P_ORANGE;
		}
		if(fld_ov < -.008 && rndpass)
		{
			color = P_VIOLET;
		}
//red-blue priority
		if(sumv != 0 && rnd & (1<<(32-10)))
		{
			color = 2-signbit(sumv);
		}
//green-yellow priority
		if(fabsf(vval) > .03 && vmj >= 0)
		{
			if(sum_yg != 0)
				color = 3 + signbit(sum_yg);
			else
				color = 3 + signbit(vval);
		}
//magneta-jade
		if(fabsf(vval) > .03 && vmj < 0)
		{
			if(sum_mj != 0)
				color = 7 + signbit(sum_mj);
			else
				color = 7 + signbit(vval);
		}
	}

	sum_yg = .02*INV_PL_VALUE*sum_yg;
	sum_mj = .02*INV_PL_VALUE*sum_mj;

	float asum3;
	if(vmj >= 0)
		asum3 = fabsf(sum_yg);
	else
		asum3 = 4*fabsf(sum_mj);

	vectField[4*ind] = (1+asum3)*vx;
	vectField[4*ind+1] = (1+asum3)*vy;
	vectField[4*ind+2] = vval;

	outval += inval_min;
	cuPhysLayerOut[ind] = outval;
	float fld = fieldLayer[NFLAYERS*ind];
	fld += sumv*(.05f*INV_PL_VALUE/(2*RSH+1)/(2*RSH+1));

	int sx, sy;
	int zx = 0;
	int zy = 0;
	float dx, dy;

	if(color <= P_BLUE || vval==0)
	{
		float lc = cuLandscape[ind];
		dx = -(lc - cuLandscape[WARPX(ix-1) + xside*(iy)]);
		dy = -(lc - cuLandscape[(ix) + xside*WARPY(iy-1)]);
	}
	else if(color <= P_YELLOW || vval || color >= P_JADE)
	{
		dx = vx;
		dy = vy;
	}
	else if(color <= P_VIOLET)
	{
		//float fld_rb = fieldLayer[NFLAYERS*ind];
		//float dxrb = fld_rb - fieldLayer[NFLAYERS*(WARPX(ix-1) + xside*iy)+1];
		//float dyrb = fld_rb - fieldLayer[NFLAYERS*(ix + xside*WARPY(iy-1))+1];

		float dx0 = fld_ov - fieldLayer[NFLAYERS*(WARPX(ix-1) + xside*iy)+2];
		float dy0 = fld_ov - fieldLayer[NFLAYERS*(ix + xside*WARPY(iy-1))+2];
		dx = dx0 - .5f*dy0;//+dxrb - .5f*dyrb;
		dy = dy0 + .5f*dx0  ;//+dyrb + .5f*dxrb;
	}

	sx = signbit(dx);
	sy = signbit(dy);
	if( color-1 )
	{
		sx = 1-sx;
		sy = 1-sy;
	}

	float adx = fabsf(dx);
	float ady = fabsf(dy);

	if((adx < DRES_INV || adx < .4f*ady) && (rnd &(1<<(32-7)))){sx = 1-sx;zx=1;}
	if((ady < DRES_INV || ady < .4f*adx) && (rnd &(1<<(32-8)))){sy = 1-sy;zy=1;}

	unsigned int sfl = signbit(fld);
	unsigned int zfl = (fld == 0)?1:0;

	if(zfl==0 && sfl == color-1 && color < 3)
	{
		zx = 0;
		zy = 0;	
	}

	int dx0 = RSH + (2*sx-1)*( ((rnd>>(32-3) & 3) >>zx) + 1);
	int dy0 = RSH + (2*sy-1)*( ((rnd>>(32-6) & 3) >>zy) + 1);


	unsigned int signs = color|(dx0<<4)|(dy0<<8);
	cuStateOut[ind] = signs;
	fieldLayer[NFLAYERS*ind] = fld;
	fieldLayer[NFLAYERS*ind+1] = sum_yg;
	fieldLayer[NFLAYERS*ind + 2] = fld_ov;
	fieldLayer[NFLAYERS*ind + 3] = sum_mj;

}

static int t_interactMatrixP[PMAX][PMAX];
static int t_interactMatrixF[PMAX][PMAX];

void ParticleLayerProc(float* cuLandscape, unsigned int* cuRand, 
					   unsigned int* cuStateIn, unsigned int* cuStateOut, float* cuPhysLayerIn,
					  float* fieldLayer, float* vectField, float* cuPhysLayerOut, float tx, float ty)
{
	memset(t_interactMatrixP, 0, sizeof(t_interactMatrixP));
	memset(t_interactMatrixF, 0, sizeof(t_interactMatrixF));

	for(int k = 0; k < MAX_INTERACTIONS; k++)
	for(int i = 0; i < MAX_INTERACTIONS; i++)
	{
		int valf = controls.interactMatrixF[k][i];
		if(valf)
		{
			t_interactMatrixF[C2P[k]][C2P[i]] = C2P[valf-1];
		}

		int valp = controls.interactMatrixP[k][i];
		if(valp)
		{
			t_interactMatrixP[C2P[k]][C2P[i]] = C2P[valp-1];
		}
	
	}

	cudaMemcpyToSymbol(interactMatrixF, t_interactMatrixF, sizeof(interactMatrixF), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(interactMatrixP, t_interactMatrixP, sizeof(interactMatrixF), 0, cudaMemcpyHostToDevice);

	int blockx = 32;
	int blocky = 32;

    dim3 grid((sim_width-2*DELTA_M)/blockx, (sim_height-2*DELTA_M)/blocky);
    dim3 block(blockx, blocky);    

	unsigned int seed = rand();
	//cudaFuncSetCacheConfig(ParticleKernel, cudaFuncCachePreferShared );

	cudaMemset(cuStateOut, 0, sim_rect*sizeof(unsigned int*));

	int shared = (2*RSH + blockx)*(2*RSH + blocky)*sizeof(PValRnd);

	ParticleKernel<<< grid, block, shared >>> (
		cuPhysLayerIn, cuStateIn, cuStateOut,
		fieldLayer,
		vectField, cuPhysLayerOut,
		cuLandscape, cuRand, sim_width, sim_height);

};
//----------------------------------------------------

__global__ void ParticleStateInitKernel(float* cuPhysLayerIn,
							   float* __restrict__ fieldLayer,
							   const float* cuLandscape, const unsigned int* cuRand,
							   unsigned int* __restrict__ cuState,
							   int xside, int yside)
{
	int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	int ind =  ix + xside*iy;

	unsigned int rnd = cuRand[ind];
	int pcolor = 1 +(rnd & 1);
	int dx0 = RSH + 1;
	int dy0 = RSH + 1;
	unsigned int signs = pcolor|(dx0<<4)|(dy0<<8);
	cuState[ind] = signs;
}

void ParticleStateInit(float* cuLandscape, unsigned int* cuRand, 
					   unsigned int* cuStateOut, float* cuPhysLayerIn,
					   float* fieldLayer)
{
	int blockx = 32;
	int blocky = 32;

    dim3 grid(sim_width/blockx, sim_height/blocky);
    dim3 block(blockx, blocky);    

	ParticleStateInitKernel<<< grid, block >>>(cuPhysLayerIn, fieldLayer, cuLandscape, cuRand,
							   cuStateOut, sim_width, sim_height);

}
#if NFLAYERS == 2
typedef float2 fltype;
#else if NFLAYERS == 4
typedef float4 fltype;
#endif

#define RDIFF 4.5f
__global__ void FieldLayerKernel(fltype* __restrict__ fieldLayerOut,
								  float4* __restrict__ vectFieldOut,
								  float invx, float invy,
							      int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	int ind = ix + xside*iy;

//-------------------------------------------------------------
	float fx0 = (ix+.5f)*invx;
	float fy0 = (iy+.5f)*invy;
//-------------------------------------------------------------	
	fltype flxm = tex2D(flTex, fx0 - RDIFF*invx, fy0);
	fltype flxp = tex2D(flTex, fx0 + RDIFF*invx, fy0);
	fltype flym = tex2D(flTex, fx0, fy0 - RDIFF*invy);
	fltype flyp = tex2D(flTex, fx0, fy0 + RDIFF*invy);
	fltype fl = tex2D(flTex, fx0, fy0);

	fltype fres;
	fres = .99f*(.6f*fl +.4f*.25f*(flxm+ flxp+ flym + flyp));

	//fres.x = .99f*(.6f*fl.x +.4f*.25f*(flxm.x + flxp.x + flym.x + flyp.x));
	//fres.y = .99f*(.6f*fl.y +.4f*.25f*(flxm.y + flxp.y + flym.y + flyp.y));
	//fres.z = .99f*(.6f*fl.z +.4f*.25f*(flxm.z + flxp.z + flym.z + flyp.z));

//-------------------------------------------------------------
	float gradflx = fl.x - flxm.x;
	float gradfly = fl.x - flym.x;

	float gradfl1x = fl.y - flxm.y;
	float gradfl1y = fl.y - flym.y;


	//float gradflwx = fl.w - flxm.w;
	//float gradflwy = fl.w - flym.w;

//fres.x < 0 red
//fres.x > 0 blue

//fres.y > 0 green
//fres.y < 0 yellow

//fres.z > 0 orange
//fres.z < 0 violet

//fres.w > 0 jade
//fres.w < 0 magneta


	if(fres.z > 0)
		fres.z = fmaxf(fres.z  - .0038f, 0);
	if(fres.z < 0)
		fres.z = fminf(fres.z  + .0038f, 0);

	float gradfl_val = gradflx*gradflx + gradfly*gradfly;

	if(interactMatrixF[P_RED][P_BLACK] == P_ORANGE)
	{
		if(gradfl_val < .005f && fres.x < 0)
		{
			fres.z += .004f;
		}
	}

	if(interactMatrixF[P_BLUE][P_BLACK] == P_VIOLET)
	{
		if(gradfl_val < .005f && fres.x > 0)
		{
			fres.z -= .004f;
		}
	}

	if(interactMatrixF[P_VIOLET][P_BLACK] == P_BLUE)
	{
		if( fres.z < 0 && fres.z > -.003)
		{
			fres.x += .1;
		}
	}

	if(interactMatrixF[P_GREEN][P_BLACK] == P_BLUE)
	{
		if( fres.y > 0 && fres.y < .003)
		{
			fres.x += .05;
		}
	}

	if(interactMatrixF[P_ORANGE][P_BLACK] == P_RED)
	{
		if( fres.z > 0 && fres.z < .003)
		{
			fres.x -= .1;
		}
	}

	if(interactMatrixF[P_YELLOW][P_BLACK] == P_RED)
	{
		if( fres.y < 0 && fres.y >- .003)
		{
			fres.x -= .05;
		}
	}


	float gradfl_val1;
	if(interactMatrixF[P_GREEN][P_BLACK] || interactMatrixF[P_YELLOW][P_BLACK])
	{
		gradfl_val1 = gradfl1x*gradfl1x + gradfl1y*gradfl1y;
	}

	if(interactMatrixF[P_GREEN][P_BLACK] == P_VIOLET)
	{
		if(gradfl_val1 < .003f && fres.y > 0)
		{
			fres.z -= .01f;
		}
	}

	if(interactMatrixF[P_YELLOW][P_BLACK] == P_ORANGE)
	{
		if(gradfl_val1 < .003f && fres.y < 0)
		{
			fres.z += .01f;
		}
	}

	fieldLayerOut[ind] = fres;
//-------------------------------------------------------------
#define LSTEP (.9f)

	fltype vvxm = tex2D(vectTex0, fx0 - 2*LSTEP*invx, fy0);
	fltype vvxp = tex2D(vectTex0, fx0 + 2*LSTEP*invx, fy0);
	fltype vvym = tex2D(vectTex0, fx0, fy0 - 2*LSTEP*invy);
	fltype vvyp = tex2D(vectTex0, fx0, fy0 + 2*LSTEP*invy);
	fltype vvc = tex2D(vectTex0, fx0, fy0);

	float4 valc = vvc + vvxm + vvxp + vvym + vvyp;


	 //float4 valc = tex2D(vectTex0, fx0, fy0)
		// + tex2D(vectTex0, fx0 - 2*LSTEP*invx, fy0)
		// + tex2D(vectTex0, fx0 + 2*LSTEP*invx, fy0)
		// + tex2D(vectTex0, fx0, fy0 - 2*LSTEP*invy)
		// + tex2D(vectTex0, fx0, fy0 + 2*LSTEP*invy);

	 float vx = valc.x;
	 float vy = valc.y;
	 float vf = valc.z;
	 float vw = valc.w;
	float gradfzx = -flxm.z + fl.z;
	float gradfzy = -flym.z + fl.z;
//----
	// float gradfwx = -flxm.w + fl.w;
	// float gradfwy = -flym.w + fl.w;
//	fres.w = .25f*(flxm.w + flxp.w + flym.w + flyp.w);// + fres.x;
	//if(fres.w > 0)
	//	fres.w -= fminf(.2f, fres.w); 
	//if(fres.w < 0)
	//	fres.w -= fmaxf(-.2f, fres.w); 


//-----
	 float l2 = vx*vx+vy*vy;
	 if(l2!=0 || interactMatrixF[P_RED][P_BLACK] == P_MAGNETA || interactMatrixF[P_BLUE][P_BLACK] == P_JADE)
	 {

	 float ln =.2*LSTEP*rsqrtf(l2);

	 vx *= ln;
	 vy *= ln;

	 float fx = fx0 + vx*invx;
	 float fy = fy0 + vy*invy;

	 float4 vres4 = tex2D(vectTex0, fx, fy);
	 float vresx = vres4.x;
	 float vresy = vres4.y;

	//if(interactMatrixF[P_BLUE][P_BLACK] == P_GREEN )
	//{
		//if(gradfl_val < .001f && fres.x > 0 && fres.x < .005)
		//{
		//	vresx += gradflx;
		//	vresy += gradfly;
		//	vf += .003;
		//}
	//}
//-------


	 //if(interactMatrixF[P_RED][P_BLACK] == P_YELLOW )
		//if(gradfl_val < .01f && fres.x < 0 && fres.x > -.005)
		//{
		//	vresx += gradflx;
		//	vresy += gradfly;
		//	vf -= .003;
		//}
//-----------

	 if(interactMatrixF[P_RED][P_BLACK] == P_MAGNETA)
		if(fres.x < 0 && fres.x > -.01)
		{
			vresx += gradflx;
			vresy += gradfly;
			vf = fminf(0, vf) - .1f;
			vw -= .01;
		}

	 if(interactMatrixF[P_BLUE][P_BLACK] == P_JADE)
		if(fres.x > 0 && fres.x < .01)
		{
			vresx -= gradflx;
			vresy -= gradfly;
			vf =  fmaxf(0, vf) + .1f;
			vw -= .01;
		}

 	float lr2 = vresx*vresx+vresy*vresy;
	if(lr2)
	{

		int vsign = (vf==0)?0:(1-2*signbit(vf));
		int vwsign = (vw==0)?0:(1-2*signbit(vw));
		float avf = fabsf(vf);

		float gox = -gradfly;
		float goy = gradflx;

		float vfield;

		if(vw >= 0)
		{
			vfield = fres.z;
			if(vsign>0)//green
			{
				vresx = .3f*gradflx  + .91f*gox;
				vresy = .3f*gradfly  + .91f*goy;
			}
			else//yellow
			{
				//orthogonal value regulate survavability	
				//- the more the stronger - .2 is indefinite, .1 is temorary
				float vresx1 = vresx - .1f*vresy - 1.5f*gradflx ;
				float vresy1 = vresy + .1f*vresx - 1.5f*gradfly ;

				vresx = vresx1;
				vresy = vresy1;
			}
		}
		else
		{
			vfield = fres.w;
			if(vsign>0)//jade
			{
				vresx = vresx - .7*gradflx - 3.5*gradfly - gox + 2*gradfzx;
				vresy = vresy - .7*gradfly + 3.5*gradflx - goy + 2*gradfzy;
				avf = max(avf - 2.9*fmaxf(fres.x, 0), 0);
			}
			else//magneta
			{
				vresx = vresx + .7*gradflx - 3.5*gradfly + gox - 2*gradfzx;
				vresy = vresy + .7*gradfly + 3.5*gradflx + goy - 2*gradfzy;
				avf = max(avf - 2.9*fmaxf(-fres.x, 0), 0);
			}
		}

		float lrs = fmaxf(.02*fabsf(vfield) + .995*sqrtf(lr2) -.0015f - 4*fmaxf(vsign*fres.z, 0), 0);//orange violet suppression
		lrs = fminf(lrs, .5f); 
		avf = fminf(avf, lrs);

//auto off jade/magneta
		//if(interactMatrixF[P_GREEN][P_BLACK] == P_BLUE)
		//{
		//	if(vf > .01)
		//	{
		//		fres.x += .01;
		//	}
		//	avf = fmaxf(avf-.02f, 0);	

		//}

		//if(interactMatrixF[P_YELLOW][P_BLACK] == P_RED)
		//{
		//	if(vf < -.01)
		//	{
		//		fres.x -= .01;
		//	}
		//	avf = fmaxf(avf-.02f, 0);				
		//}

		float lrnorm = avf*rsqrtf(vresx*vresx+vresy*vresy);
		vresx *= lrnorm;
		vresy *= lrnorm;
		float vf1 = avf*vsign;
	
		vectFieldOut[ind].x = vresx;
		vectFieldOut[ind].y = vresy;
		vectFieldOut[ind].z = vf1;
		vectFieldOut[ind].w= avf*vwsign;
	 }
	 }//l2 != 0

//	fieldLayerOut[ind] = fres;

}
		
extern cudaArray *gCudaVectArray;
extern cudaArray *gCudaFlArray;
extern cudaArray *gCudaFuncSmooth;
void InitVfield(float* vectField, int xside, int yside);

void FieldLayerProc(float* cuLandscape,
					float* fieldLayerIn, float* fieldLayerOut,
					float* vectField, int xside, int yside)
{

	int blockx = 64;
	int blocky = 8;

	cudaArray * arrV = gCudaVectArray;

	cudaMemcpy2DToArray(arrV, 0, 0, vectField, 4*xside*sizeof(float), 4*xside*sizeof(float), yside, cudaMemcpyDeviceToDevice);
	cudaMemset2D(vectField, 4*xside*sizeof(float), 0, 4*xside*sizeof(float), yside);

    struct cudaChannelFormatDesc descV; 
    cutilSafeCall(cudaGetChannelDesc(&descV, arrV));
	vectTex0.addressMode[0] = cudaAddressModeWrap;
	vectTex0.addressMode[1] = cudaAddressModeWrap;
	vectTex0.filterMode = cudaFilterModeLinear;
	vectTex0.normalized = true;
    cutilSafeCall(cudaBindTextureToArray(&vectTex0, arrV, &descV));

	cudaArray * arrF = gCudaFlArray;
	cudaMemcpy2DToArray(arrF, 0, 0, fieldLayerIn, NFLAYERS*xside*sizeof(float), NFLAYERS*xside*sizeof(float), yside, cudaMemcpyDeviceToDevice);
	//cudaMemset2D(fieldLayerIn, NFLAYERS*xside*sizeof(float), 0, NFLAYERS*xside*sizeof(float), yside);

    struct cudaChannelFormatDesc descF; 
    cutilSafeCall(cudaGetChannelDesc(&descF, arrF));
	flTex.addressMode[0] = cudaAddressModeWrap;
	flTex.addressMode[1] = cudaAddressModeWrap;
	flTex.filterMode = cudaFilterModeLinear;
	flTex.normalized = true;
    cutilSafeCall(cudaBindTextureToArray(&flTex, arrF, &descF));

	cudaArray * arr1d = gCudaFuncSmooth;
    struct cudaChannelFormatDesc desc1; 
    cutilSafeCall(cudaGetChannelDesc(&desc1, arr1d));
	smoothFunc.addressMode[0] = cudaAddressModeClamp;
	smoothFunc.filterMode = cudaFilterModeLinear;
	smoothFunc.normalized = true;
    cutilSafeCall(cudaBindTextureToArray(&smoothFunc, arr1d, &desc1));
	
	cudaFuncSetCacheConfig(FieldLayerKernel, cudaFuncCachePreferL1);
	
    dim3 grid(sim_width/blockx, sim_height/blocky);
    dim3 block(blockx, blocky);    
	FieldLayerKernel<<< grid, block >>> (
		(fltype*)fieldLayerOut,
		(float4*)vectField, 
		1.f/sim_width, 1.f/sim_height,
		sim_width, sim_height);

	cudaUnbindTexture(vectTex0);
	cudaUnbindTexture(flTex);

};