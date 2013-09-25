#ifndef _BHV_H_
#define _BHV_H_

#define MAX_SET_SIZE 16
#define MAX_DECK 256
#define MAX_BONE_ELEMNTS 256

struct Bone
{
int src; int connect; int dst;
};


class BhvEelemntAbstract
{
public:
	float score;
	Bone boneId;
	virtual float ConditionScore(){return 0;};
	virtual void Process(){};
	virtual void Tick(){};
	virtual void Render(int x, int y){};
};

struct BoneSet
{
	BhvEelemntAbstract* bhv_ptr;
	int num;
};

struct ActiveSets
{
	int blueSetSize;
	int redSetSize;
	float statusPickRed;
	float statusPickBlue;

	BoneSet blueBoneSet[MAX_SET_SIZE]; 
	BoneSet redBoneSet[MAX_SET_SIZE];
	ActiveSets()
	{
		statusPickRed = 0;
		statusPickBlue = 0;
		blueSetSize =0;
		redSetSize = 0;
		memset(blueBoneSet, 0, MAX_SET_SIZE*sizeof(BoneSet));
		memset(redBoneSet, 0, MAX_SET_SIZE*sizeof(BoneSet));
	};
	void Pick();
	void AddRedBone(BhvEelemntAbstract* bhv_ptr);
	void AddBlueBone(BhvEelemntAbstract* bhv_ptr);

};

extern ActiveSets gActiveSets;

//------------------------------------------------------
//heuristics 
extern GSum gFiledIntegral;


class BhvBoneEelemnt:public BhvEelemntAbstract
{
public:
	BhvBoneEelemnt(){memset(&boneId, 0, sizeof(boneId));}
	float ConditionScore(){return 0;};
	virtual void Process(){};
	virtual void Tick(){};
	virtual void Render(int x, int y){};
};
//-----------------------------------------

class LinkElement:public BhvBoneEelemnt 
{
public:
	LinkElement(){
		score = 0;
		memset(&boneId, 0, sizeof(boneId));
	};

	LinkElement(int src, int conn, int dst)
	{
		SetBone(src, conn, dst);
	};

	void SetBone(int src, int conn, int dst)
	{
		boneId.src =src;
		boneId.connect =conn;
		boneId.dst = dst;
	};
	
	float ConditionScore(){
		return 0;
	};

	void Render(int x, int y);
	void Process();
};

//-----------------------------------------
//blue elements

//links
class GreenVioletHeuristic:public LinkElement //green penetration
{
public:
	GreenVioletHeuristic():LinkElement(P_GREEN, P_VIOLET, P_BLACK){
		score = 1;
	};
	float ConditionScore();
};

class BlueVioletHeuristic:public LinkElement 
{
public:
	BlueVioletHeuristic():LinkElement(P_BLUE, P_VIOLET, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};

class VioletBlueHeuristic:public LinkElement 
{
public:
	VioletBlueHeuristic():LinkElement(P_VIOLET, P_BLUE, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};

class GreenBlueHeuristic:public LinkElement 
{
public:
	GreenBlueHeuristic():LinkElement(P_GREEN, P_BLUE, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};


class BlueJadeHeuristic:public LinkElement 
{
public:
	BlueJadeHeuristic():LinkElement(P_BLUE, P_JADE, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};

class BlueBlackHeuristic:public LinkElement 
{
public:
	BlueBlackHeuristic():LinkElement(P_BLUE, P_BLACK, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};

class GreenBlackHeuristic:public LinkElement 
{
public:
	GreenBlackHeuristic():LinkElement(P_GREEN, P_BLACK, P_BLACK){
		score = 1;
	};
	float ConditionScore();

};


//-------------

class GreenBlobHeuristic:public BhvBoneEelemnt
{
public:
	GreenBlobHeuristic(){
		score = 10;
		memset(&boneId, 0, sizeof(boneId));
		boneId.src = -1;
		boneId.dst = P_GREEN;
	};

	float ConditionScore(){
		return 20;
	};

	void Render(int x, int y);
	void Process();
};

class JadeRingHeuristic:public BhvBoneEelemnt
{
public:
	JadeRingHeuristic(){
		score = 9;
		memset(&boneId, 0, sizeof(boneId));
		boneId.src = -1;
		boneId.dst = P_JADE;
	};

	float ConditionScore();

	void Render(int x, int y);
	void Process();
};
//----------------------------
//elements are not heuristics, but could be made into them later
//red elements


class YellowBlobElement:public BhvBoneEelemnt
{
public:
	YellowBlobElement(){
		score = 1;
		memset(&boneId, 0, sizeof(boneId));
		boneId.src = -1;
		boneId.dst = P_YELLOW;
	};

	float ConditionScore(){
		return 1;
	};

	void Render(int x, int y);
	void Process();
};

class MagnetaRingElement:public BhvBoneEelemnt
{
public:
	MagnetaRingElement(){
		score = 1;
		memset(&boneId, 0, sizeof(boneId));
		boneId.src = -1;
		boneId.dst = P_MAGNETA;
	};

	float ConditionScore(){
		return 1;
	};

	void Render(int x, int y);
	void Process();
};

extern GreenVioletHeuristic greenVioletElement;
extern BlueVioletHeuristic blueVioletElement;
extern VioletBlueHeuristic violetBlueElement;
extern GreenBlueHeuristic greenBlueElement;
extern BlueJadeHeuristic blueJadeElement;

extern GreenBlobHeuristic greenBlobElement;
extern JadeRingHeuristic jadeRingElement;

extern YellowBlobElement yellowBlobElement;
extern MagnetaRingElement magnetaRingElement;

struct ObjInertia
{
	void Init(float2 pos0, float2 pos1)
	{
		spos0 = 0;
		spos1 = 0;

		vp0.x = 0;
		vp0.y = 0;

		vp1.x = 0;
		vp1.y = 0;

		atten0 = 1;
		atten1 = 1;
		for(int i =0; i < STACK_TSIZE; i++)
		{
			stackTPos0[i] = pos0;
			stackTPos1[i] = pos1;
		}

	};

	void Step(float tx, float ty, float tx1, float ty1);

	int spos0;
	int spos1;
	float2 stackTPos0[STACK_TSIZE];
	float2 stackTPos1[STACK_TSIZE];
	float2 vp0;
	float2 vp1;
	float atten0;
	float atten1;
};
#define MIN_SET_SIZE 3
#endif