// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h


// Shared Library Test Functions
#include <shrUtils.h>
#include "layer_defines.h"

#include "bhv.h"


//------------------------------------------------------

//move to singleton
 
float gIntegralScale = 160000;

//------------------------------------------------------

void ReadRedBoneSet();
void ReadBlueBoneSet();
void InitRedBoneBhv();
void InitBlueBoneBhv();

void InitBhv()
{
	InitBlueBoneBhv();
	InitRedBoneBhv();
// init
	ReadBlueBoneSet();
	ReadRedBoneSet();
};

void ObjBhv();
int CardsBhv(BoneSet* aiBoneSet, int &aiSetSize);

extern GSum gFiledIntegral;

bool BlueSetTest(int src, int conn, int dest);
bool RedSetTest(int src, int conn, int dest);
extern BhvEelemntAbstract* gElementPtr[MAX_BONE_ELEMNTS];
extern int startBlue;
extern int sizeBlue;
extern int startRed;
extern int sizeRed;

void DoBhv()
{
	ObjBhv();
	
//return;
//assume ai is blue
	if(gActiveSets.statusPickBlue >= 1)
	{
		if(gActiveSets.blueSetSize <= MIN_SET_SIZE)
		{
			bool green_present = BlueSetTest(-1, 0, P_GREEN);
			bool jade_present = BlueSetTest(-1, 0, P_JADE);
			if(!green_present && !jade_present)
			{
				if(rand()%2)
					gActiveSets.AddBlueBone(&jadeRingElement);
				else
					gActiveSets.AddBlueBone(&greenBlobElement);
			}
			else
			{
				int ind = startBlue + rand()%sizeBlue;
				gActiveSets.AddBlueBone(gElementPtr[ind]);
			}

		}
		else if(CardsBhv(gActiveSets.blueBoneSet, gActiveSets.blueSetSize) >= 0)
			gActiveSets.statusPickBlue -= 1;
	}

	if(gActiveSets.statusPickRed >= 1)
	{
		if(gActiveSets.redSetSize <= MIN_SET_SIZE)
		{
			bool yellow_present = RedSetTest(-1, 0, P_YELLOW);
			bool magenta_present = RedSetTest(-1, 0, P_MAGNETA);
			if(!yellow_present && !magenta_present)
			{
				if(rand()%2)
					gActiveSets.AddRedBone(&magnetaRingElement);
				else
					gActiveSets.AddRedBone(&yellowBlobElement);
			}
			else
			{
				int ind = startBlue + rand()%sizeRed;
				gActiveSets.AddRedBone(gElementPtr[ind]);
			}

		}
		else if(CardsBhv(gActiveSets.redBoneSet, gActiveSets.redSetSize) >= 0)
			gActiveSets.statusPickRed -= 1;
	}

}

void ArrayCleanup(BoneSet* boneSet, int& setSize);

int CardsBhv(BoneSet* aiBoneSet, int &aiSetSize)
{
	//find max score; should be probability instead of max
	float max_score = 0;
	int indElem = -1;
	//could be several passes for iterative method
	for (int i =0; i < aiSetSize; i++)
	{
		if(aiBoneSet[i].num <= 0)
			continue;

		BhvEelemntAbstract* element = aiBoneSet[i].bhv_ptr;
		float score = element->ConditionScore();
		if(score > max_score)
		{
			max_score = score;
			indElem = i;
		}
		element->Tick();//all elemnts are thinking meanwhile
	}
	if(indElem >= 0)
	{
		BhvEelemntAbstract* element = aiBoneSet[indElem].bhv_ptr;
		element->Process();
		aiBoneSet[indElem].num--;
		ArrayCleanup(aiBoneSet, aiSetSize);
		return 0;
	}
	return -1;
}

void ArrayCleanup(BoneSet* boneSet, int& setSize)
{
	int krem = -1;

	do
	{
		krem = -1;
		for (int i =0; i < setSize; i++)
		{
			if(boneSet[i].num <= 0)
			{
				krem = i;
				break;
			}
		}
		if(krem >= 0)
		{
			for(int i = krem; i < setSize-1; i++)
				boneSet[i] = boneSet[i+1];
			setSize--;
		}
	}while(setSize >0 && krem >= 0);
}


void vectPatch( float* vectField, float cx, float cy, float r, int type,
								float vx, float vy, int xside, int yside);
extern float gObj1X;
extern float gObj1Y;
extern TCMaxHost mmGrid[256*256];
extern int mmGridSize;
extern int mmGridSizeX;
extern int mmGridSizeY;
extern float *gVectorLayer;
extern float threshold_YG;
extern float* grid8ValTick;

void GreenBlobHeuristic::Process()
{
	int me = ME_BLUE;
	int tag = ME_RED;
	int sign =(me==ME_RED)?1:-1;

//make avergae and random

	int num = 0;
	float avg =0;
	for(int by = 0; by < mmGridSizeY; by++)
	for(int bx = 0; bx < mmGridSizeX; bx++)
	{
		int ind = bx + by*mmGridSizeX;
		if(mmGrid[ind].ind[tag] < 0)
			continue;
		float vc = sign*mmGrid[ind].v[tag];
		if(vc>0)
			avg += vc;
		num++;
	}
	if(num == 0)
		return;

	avg /= num;

	int ind_s = -1;
	float val_tag = 0;

	for(int by = 0; by < mmGridSizeY; by++)
	for(int bx = 0; bx < mmGridSizeX; bx++)
	{
		int ind = bx + by*mmGridSizeX;
		if(mmGrid[ind].ind[tag] < 0)
			continue;
		float vc = sign*mmGrid[ind].v[tag];
		float rndv = rand()*1.f/RAND_MAX;
		vc += rndv*.1f*avg;
		if(vc > val_tag)
		{
			val_tag = vc;
			ind_s = mmGrid[ind].ind[tag];
		}
	}

	int iyt = ind_s/sim_width;
	int ixt =  ind_s - iyt*sim_width;


	float a = (float)(rand()%360);

	float vx = .5f + .5f*cos(2*M_PI*a/360);
	float vy = .5f + .5f*sin(2*M_PI*a/360);
	int rad = 20;
	int type = 0;
	vectPatch(gVectorLayer, (float)ixt, (float)iyt, (float)rad, type,
					vx,  vy, sim_width, sim_height);

}

int GetYGPos(int& resx, int& resy, int ygind);

void JadeRingHeuristic::Process()
{
	float fxt = gObj1X;
	float fyt = gObj1Y;
	
	int ixt, iyt;
	int res = GetYGPos(ixt, iyt, 1);// check for yellow

	if(res >= 0)
	{
		fxt = (float)ixt;
		fyt = (float)iyt;
	}


	float a = (float)(rand()%360);

	float vx = .5f + .5f*cos(2*M_PI*a/360);
	float vy = .5f + .5f*sin(2*M_PI*a/360);
	int rad = 20;
	int type = 2;
	vectPatch(gVectorLayer, fxt, fyt, (float)rad, type,
					vx,  vy, sim_width, sim_height);

}
#include "glcontrols.h"

void LinkElement::Process()
{
	int newVal =  P2C[boneId.connect] + 1;

	if(P2C[boneId.connect] == BLACK)
		newVal = 0;

	controls.interactMatrixF[P2C[boneId.src]][P2C[boneId.dst]] = newVal;
}
extern int pickState;
void YellowBlobElement::Process()
{
	pickState = 2;
}

void MagnetaRingElement::Process()
{
	pickState = 4;
}

bool RedSetTest(int src, int conn, int dest)
{

	for(int i = 0; i < gActiveSets.redSetSize; i++)
	{
		if(gActiveSets.redBoneSet[i].bhv_ptr->boneId.src == src &&
			gActiveSets.redBoneSet[i].bhv_ptr->boneId.connect == conn &&
			gActiveSets.redBoneSet[i].bhv_ptr->boneId.dst == dest)
				return true;
	}

	return false;
}

bool HexLinkTest(int src, int conn, int dest)
{
	if(controls.interactMatrixF[P2C[src]][P2C[dest]] == P2C[conn] + 1)
		return true;

	return false;
}

bool BlueSetTest(int src, int conn, int dest)
{

	for(int i = 0; i < gActiveSets.blueSetSize; i++)
	{
		if(gActiveSets.blueBoneSet[i].bhv_ptr->boneId.src == src &&
			gActiveSets.blueBoneSet[i].bhv_ptr->boneId.connect == conn &&
			gActiveSets.blueBoneSet[i].bhv_ptr->boneId.dst == dest)
				return true;
	}

	return false;
}

float JadeRingHeuristic::ConditionScore()
{
	bool yellow_present = gFiledIntegral.v[P_YELLOW] > 10;

	if(yellow_present)
		return 25;//debug

	if(rand()%3)
		return 1;

	return 0;
}

float GreenVioletHeuristic::ConditionScore(){
	bool orange_present = gFiledIntegral.v[P_ORANGE] > 10
		||  RedSetTest(P_RED, P_ORANGE, P_BLACK);

	bool green_present = gFiledIntegral.v[P_GREEN] > 10 ||
		BlueSetTest(-1, 0, P_GREEN);

	if(orange_present && green_present)
		return 5;
	
	return 0;
};

float BlueVioletHeuristic::ConditionScore(){
	bool yellow_present = gFiledIntegral.v[P_YELLOW] > 10
		||  RedSetTest(-1, 0, YELLOW);

	if(yellow_present)
		return 6;
	
	return 0;
};

float VioletBlueHeuristic::ConditionScore(){
	bool violet_present = gFiledIntegral.v[P_VIOLET] > 1000 || HexLinkTest(P_BLUE, P_VIOLET, P_BLACK);;

	if(violet_present)
		return 20;
	
	return 0;
};

float GreenBlueHeuristic::ConditionScore(){

	bool green_present = gFiledIntegral.v[P_GREEN] > 2000;

	float red = gFiledIntegral.v[P_RED]/gIntegralScale;

	if(red < .3 || green_present)
		return 4;
	
	return 0;
};

float BlueJadeHeuristic::ConditionScore(){

	bool green_present = gFiledIntegral.v[P_GREEN] > 2000;
	bool greenBone = BlueSetTest(-1, 0, P_GREEN);
	bool magneta_present = gFiledIntegral.v[P_MAGNETA] > 100;

	if(magneta_present)
		return 10;

	if(green_present)
		return 3;

	if(greenBone)
		return .5f;

	return 0;
	
};

float BlueBlackHeuristic::ConditionScore(){

	float blue = gFiledIntegral.v[P_BLUE]/gIntegralScale;

	bool yellow_present = gFiledIntegral.v[P_YELLOW] > 2000; 

	bool yellow_bone = RedSetTest(-1, 0, P_YELLOW);

	bool orange_present = gFiledIntegral.v[P_ORANGE] > 3000; 

	bool orangeBone = BlueSetTest(P_BLUE,  P_ORANGE, P_BLACK);

	if(yellow_present || !orange_present)
		return 0;

	if(yellow_bone && orangeBone && blue < .3)
		return 2.f;

	return 0;
	
};

float GreenBlackHeuristic::ConditionScore(){

	bool green_present = gFiledIntegral.v[P_GREEN] > 2000;
	bool greenBone = BlueSetTest(-1, 0, P_GREEN);

	float red = gFiledIntegral.v[P_RED]/gIntegralScale;
	float blue = gFiledIntegral.v[P_BLUE]/gIntegralScale;


	if(green_present && greenBone && red < .3)
		return 4;



	return 0;
	
};