// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

// Shared Library Test Functions
#include <shrUtils.h>
#include "layer_defines.h"

#include "bhv.h"

int globalDataPeriod = 2;
extern float *tempHostDataNoCuda;
extern TCMaxHost mmGrid[256*256];
extern float* grid8ValTick;
extern int mmGridSize;
extern int mmGridSizeX;
extern int mmGridSizeY;


#define DEFAULT_OBJ_BHV 0
#define ACT_OBJ_BHV_MOVE 1
#define ACT_OBJ_BHV_JUMP 2
#define ACT_OBJ_BHV_STAY 3

class LObjectBhv:public BhvEelemntAbstract
{
public:
	LObjectBhv()
	{
		objBhvCounter = 0;
		state = DEFAULT_OBJ_BHV;
		xme = -1;
		yme = -1;
		xtarget = -1;
		ytarget = -1;
		me = ME_BLUE;
		target = ME_RED;
		width = 128;
		memset(startVal, 0, sizeof(startVal));
	};
	int objBhvCounter;
	int state;
	void Tick();
	void CheckStay();
	bool CheckMove();
	bool CheckJump();
	void Move();
	void Jump();
	int xtarget, ytarget;
	int xme, yme;
	int me, target;
	int width;
	float startVal[4];
};


//#define RUN_PERIOD 100
#define RUN_PERIOD 10
#define ACT_PERIOD 2

extern float gObj1X;
extern float gObj1Y;

void LObjectBhv::Tick()
{
	objBhvCounter++;

	if( objBhvCounter >= RUN_PERIOD || state == DEFAULT_OBJ_BHV)
	{

		for(int by = 0; by < mmGridSizeY; by++)
		for(int bx = 0; bx < mmGridSizeX; bx++)
		{
			int ind = bx + by*mmGridSizeX;
			grid8ValTick[8*ind] = -1;
			grid8ValTick[8*ind+1] = -1;
		}
		
		if(state == DEFAULT_OBJ_BHV)
		{
			xme =(int)gObj1X;//temporary!
			yme = (int)gObj1Y;
		}

		CheckStay();
		objBhvCounter = min(objBhvCounter, objBhvCounter/2);

		if(state == ACT_OBJ_BHV_STAY)
			objBhvCounter =0;


		if(state == ACT_OBJ_BHV_MOVE)
		{
			if(CheckMove())
				objBhvCounter =0;
			else
				state = ACT_OBJ_BHV_JUMP;

		}

		if(state == ACT_OBJ_BHV_JUMP)
		{
			if(CheckJump())
				objBhvCounter =0;
			else
				state = DEFAULT_OBJ_BHV;

		}
		
		if(state == ACT_OBJ_BHV_JUMP)
		{
			Jump();
			state = ACT_OBJ_BHV_STAY;
		}
	
	}

	if(objBhvCounter%ACT_PERIOD == 0 && state == ACT_OBJ_BHV_MOVE)
		Move();
};

class LSum;
extern float *gRedBlueField;
extern float* cuTempData;
int localSum(int xc, int yc, int width, int xside, int yside, float4 *field, LSum* cuTempData, float* fsum);


void GetFiledIntegrals(float* v, GSum& integral);
void GetPoints(int xs, int ys, int xe, int ye, int size, float* arr);
void GetPoint(int ix, int iy, float* valArr);

#define LINES 5
static float arrline[4*LINES];


extern float *tempHostData;
extern float *tempHostDataNoCuda;

float debugArr[256*256*8];

float threshold_YG = 100;
float scale_red = 1.f;
float admissible_ratio = 2.f;
float favorable_ratio = .7f;
float min_opposition_ratio = .2f;
//act
float thresholdMove = .8f;
float deltaPointCheck = 10;
int max_block_move = 8;


void LObjectBhv::CheckStay()
{
//debug
//return;


	float fsum_raw[8];
	int ret = localSum(xme, yme, width, sim_width, sim_height, (float4*)gRedBlueField, (LSum*)cuTempData, fsum_raw);

	if(ret < 0)
	{
		state = ACT_OBJ_BHV_JUMP;
		return;
	}

	GSum local_integral;
	GetFiledIntegrals(fsum_raw, local_integral);

	state = ACT_OBJ_BHV_STAY;

	if(me == ME_BLUE)
	{
		if(local_integral.v[P_YELLOW] > threshold_YG)
			state = ACT_OBJ_BHV_JUMP;
		if(admissible_ratio*local_integral.v[P_BLUE] < scale_red*local_integral.v[P_RED])
			state = ACT_OBJ_BHV_JUMP;
		if(min_opposition_ratio*local_integral.v[P_BLUE] >  local_integral.v[P_RED])
			state = ACT_OBJ_BHV_MOVE;
	}
	else
	{
		if(local_integral.v[P_GREEN] > threshold_YG)
			state = ACT_OBJ_BHV_JUMP;
		if(admissible_ratio*scale_red*local_integral.v[P_RED] < local_integral.v[P_BLUE])
			state = ACT_OBJ_BHV_JUMP;
		if(min_opposition_ratio*local_integral.v[P_RED] >  local_integral.v[P_BLUE])
			state = ACT_OBJ_BHV_MOVE;
	}
}

int deb_flag=0;
int deb_flag1=0;

void GetPoints(int xs, int ys, int xe, int ye, int size, float* arr);
extern int blockSizex;
extern int blockSizey;
int tempMask[512*512];

bool LObjectBhv::CheckMove()
{

	GetPoint(xme, yme, startVal);

	memset(tempMask, 0, mmGridSizeY*mmGridSizeX*sizeof(int));

	int gap = 32;
	int ming = 64;

	int bcx = mmGridSizeX/2;
	int bcy = mmGridSizeY/2;
	int minW = min(mmGridSizeX/2, mmGridSizeY/2);
	int startr = 0;
	if(state == ACT_OBJ_BHV_MOVE)
	{
		bcx = xme/blockSizex;
		bcy = yme/blockSizey;
		startr = 1;
	}

	int itag = -1;

	for(int rk0 = startr; rk0 <= max_block_move; rk0++)
	{
	for(int dy0 = -rk0; dy0 <= rk0; dy0++)
	{
	for(int dx0 = -rk0; dx0 <= rk0; dx0++)
	{
		if(abs(dx0) != rk0 && abs(dy0 != rk0))
			continue;
		
		int bx = (bcx+dx0+mmGridSizeX)%mmGridSizeX;	
		int by = (bcy+dy0+mmGridSizeY)%mmGridSizeY;	

		int ind_tag = bx + by*mmGridSizeX;
		if(mmGrid[ind_tag].ind[target] < 0)
			continue;

		if(tempMask[ind_tag])
			continue;

		tempMask[ind_tag] = 1;

		float av = fabs(mmGrid[ind_tag].v[target]);
		int iyt = mmGrid[ind_tag].ind[target]/sim_width;
		int ixt =  mmGrid[ind_tag].ind[target] - iyt*sim_width;

		int ret =0;
		if(grid8ValTick[8*ind_tag] < 0)
		{
			float fsum_raw[8];
			ret = localSum(ixt, iyt, width, sim_width, sim_height, (float4*)gRedBlueField, (LSum*)cuTempData, fsum_raw);
			memcpy(grid8ValTick + 8*ind_tag, fsum_raw, 8*sizeof(float));

		}
		
		if(ret < 0)
			continue;

			
		bool fail = false;

		if(me == ME_BLUE)
		{
			if(grid8ValTick[8*ind_tag+ME_YELLOW] > threshold_YG)
				fail = true;
			//if(admissible_ratio*grid8ValTick[8*ind_tag+ME_BLUE] < scale_red*grid8ValTick[8*ind_tag+ME_RED])
			//	fail = true;
			if(min_opposition_ratio*grid8ValTick[8*ind_tag+ME_BLUE] >  grid8ValTick[8*ind_tag+ME_RED])
				fail = true;

		}
		else
		{
			if(grid8ValTick[8*ind_tag+ME_GREEN] > threshold_YG)
				fail = true;
			//if(admissible_ratio*scale_red*grid8ValTick[8*ind_tag+ME_RED] < grid8ValTick[8*ind_tag+ME_BLUE])
			//	fail = true;
			if(min_opposition_ratio*grid8ValTick[8*ind_tag+ME_RED] >  grid8ValTick[8*ind_tag+ME_BLUE])
				fail = true;
		}

		if(fail)
			continue;
		else
		{
			itag = ind_tag;
			break;
		}

	}
		if(itag>=0)
			break;
	}
		if(itag>=0)
			break;
	}//end check for move

//make it separtae function?

	if(itag < 0)
		return false;

	ytarget =  mmGrid[itag].ind[target]/sim_width;
	xtarget =  mmGrid[itag].ind[target] - ytarget*sim_width;

//	assert(xtarget < sim_width	&& ytarget < sim_height);

	return true;
};

bool LObjectBhv::CheckJump()
{
	float fitness = 1e10;
	int itag = -1;
	int ime = -1;

	memset(tempMask, 0, mmGridSizeY*mmGridSizeX*sizeof(int));
		
	int maxrk = max(mmGridSizeX/2, mmGridSizeY/2);

	int bxold =-1;
	int byold= -1;
	if(xme >= 0 && yme >= 0)
	{
		//skip old
		bxold =  xme/blockSizex;
		byold =  yme/blockSizex;
	}

	ime = -1;

//check for jump move
	for(int by = 0; by < mmGridSizeY; by++)
	for(int bx = 0; bx < mmGridSizeX; bx++)
	{
		int ind = bx + by*mmGridSizeX;
		if(mmGrid[ind].ind[me] < 0)
			continue;

		int iym = mmGrid[ind].ind[me]/sim_width;
		int ixm =  mmGrid[ind].ind[me] - iym*sim_width;

		int ret =0;
		if(grid8ValTick[8*ind] < 0)//validity test
		{
			float fsum_raw[8];
			ret = localSum(ixm, iym, width, sim_width, sim_height, (float4*)gRedBlueField, (LSum*)cuTempData, fsum_raw);
			memcpy(grid8ValTick + 8*ind, fsum_raw, 8*sizeof(float));
		}
		if(ret < 0)
		{
			if(rand()/RAND_MAX <= .5)
				continue;
			else
			{
				bool stop = false;
				for(int rk0 = 1; rk0 <= max_block_move; rk0++)
				{
					for(int dy0 = -rk0; dy0 <= rk0; dy0++)
					{
						for(int dx0 = -rk0; dx0 <= rk0; dx0++)
						{
							if(abs(dx0) != rk0 && abs(dy0 != rk0))
								continue;
							
							int bnx = bx+dx0;	
							int bny = by+dy0;	

							if(bnx < 0 || bnx >= mmGridSizeX)
								continue;

							if(bny < 0 || bny >= mmGridSizeY)
								continue;

							int indn = bnx + bny*mmGridSizeX;
							if(grid8ValTick[8*indn] < 0)
								continue;
							memcpy(grid8ValTick + 8*ind, grid8ValTick + 8*indn, 8*sizeof(float));
							stop = true;
							break;
						}
						if(stop) break;
					}
					if(stop) break;
				}
				if(!stop)	continue;
			}
		}

		GSum local_integral;
		GetFiledIntegrals(grid8ValTick + 8*ind, local_integral);
		float av = 0, aopp = 0; 
		
		if(me == ME_BLUE)
		{
			av = local_integral.v[P_BLUE];
			aopp = local_integral.v[P_RED];
			if(local_integral.v[P_YELLOW] > threshold_YG)
				continue;
		}
		else
		{
			av = local_integral.v[P_RED];
			aopp = local_integral.v[P_BLUE];
			if(local_integral.v[P_GREEN] > threshold_YG)
				continue;
		}

		float lfitness = fabs(favorable_ratio*av - aopp);
		if(bx == bxold && by == byold)
			lfitness += 10;
//ranomize
		float rnd_add = rand()*1.f/RAND_MAX;
		lfitness += rnd_add*.005f*favorable_ratio*av;

		if(lfitness < fitness)
		{
			xme = ixm;
			yme = iym;
			ime = ind;
			fitness = lfitness;
		}		

	}

	if(ime >= 0)
		return true;
	else
		return false;
};


void LObjectBhv::Jump()
{
	gObj1X = (float)xme;
	gObj1Y = (float)yme;
};

void LObjectBhv::Move()
{
	int kk_target = 1;
	int kk_me = 0;
	int mysign = 1-2*kk_me;

//	integral.v[P_RED] = v[1];
//	integral.v[P_BLUE] = v[0];
	int size = LINES;


	float dx = (float)xtarget - xme;
	float dy = (float)ytarget - yme;
	if(fabsf(dx) <  deltaPointCheck/size && fabsf(dy) <  deltaPointCheck/size)
	{
		state = ACT_OBJ_BHV_STAY;
		return;
	}

	float ilen = deltaPointCheck/sqrt(dx*dx + dy*dy);
	dx *= ilen;
	dy *= ilen;
	int x1 = (int)(xme + dx);
	int y1 = (int)(yme + dy);
	GetPoints(xme, yme, x1, y1, LINES, arrline);
	float vs = arrline[kk_me];
	float ve = arrline[4*(size-1) + kk_me];

	float threshold = startVal[kk_me]*thresholdMove*mysign;

	if(vs*mysign < threshold)
	{
		state = ACT_OBJ_BHV_STAY;
		return;
	}
	
	int ithr = size-1;
	float deltas =0;
	for(int i=1; i < size; i++)
	{
		float floatv = arrline[4*i+kk_me];
		if(floatv*mysign < threshold)
		{
			float deltai = mysign*(arrline[4*(i-1)+kk_me] - floatv);
			float delta0 = mysign*(arrline[4*(i-1)+kk_me] - threshold);
			deltas = delta0/deltai;
			ithr = i-1;
			break;
		}
	}

	if(ithr < 1)
	{
		state = ACT_OBJ_BHV_STAY;
		return;
	}

	xme = (int)(xme + (ithr+deltas)*dx/size);
	yme = (int)(yme + (ithr+deltas)*dy/size);

	Jump();

};

LObjectBhv shamTumb;

void ObjBhv()
{
	shamTumb.Tick();
};