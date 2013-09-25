#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "layer_defines.h"
#include "bhv.h"
extern ActiveSets gActiveSets;
extern BhvEelemntAbstract* gElementPtr[MAX_BONE_ELEMNTS];
extern int gNumElements;

BhvEelemntAbstract* FindElement(int src, int connect, int dst)
{

	return NULL;
}

char* colorNames[] =
{
"RED",
"YELLOW",
"VIOLET",
"BLUE",
"GREEN",
"ORANGE",
"MAGNETA",
"JADE",
"BLACK"
};
extern int C2P[];
int getColor(char* name)
{
	for(int i=0; i < MAX_INTERACTIONS; i++)
	{
		if(strcmp(name, colorNames[i]) == 0)
			return i;
	}

	return -1;
}

extern LinkElement redLinkElement[MAX_DECK];
extern YellowBlobElement yellowBlobElement;
extern MagnetaRingElement magnetaRingElement;

void readStack(char* fileName, int rb)
{
	FILE* fr = fopen(fileName, "rt");
	if(fr == NULL)
		printf("Error: bone stack file %s not ound \n", fileName);

	char line[128];
	char s1[128];
	char s2[128];
	char s3[128];
	while(fgets(line, 128, fr))
	{
		int nread = sscanf(line, "%s %s %s", s1, s2, s3);

		int src = getColor(s1);
		int connect = getColor(s2);
		int dst = getColor(s3); 

		if(src < 0 || connect < 0 || dst < 0)
		{
			printf("stack file read error: %s %s %s \n", s1, s2, s3);
			continue;
		}
		src = C2P[src];
		connect = C2P[connect];
		dst =C2P[dst]; 



		if(connect == P_BLACK && src == P_BLACK)
		{
			if(rb == P_RED)
			{
				if(dst == P_YELLOW)
				{
					gActiveSets.AddRedBone(&yellowBlobElement);
				}
				else if(dst == P_MAGNETA)
				{
					gActiveSets.AddRedBone(&magnetaRingElement);
				}

			}
			else if(rb == P_BLUE)
			{
				if(dst == P_GREEN)
				{
					gActiveSets.AddBlueBone(&greenBlobElement);
				}
				else if(dst == P_JADE)
				{
					gActiveSets.AddBlueBone(&jadeRingElement);
				}			
			}
		}
		else
		{
			int kf = -1;
			for(int i = 0; i < gNumElements; i++)
			{
				if(gElementPtr[i]->boneId.src == src && gElementPtr[i]->boneId.dst == dst && gElementPtr[i]->boneId.connect == connect)
				{
					if(rb == P_RED)
						gActiveSets.AddRedBone(gElementPtr[i]);
					else if(rb == P_BLUE)
						gActiveSets.AddBlueBone(gElementPtr[i]);
					break;
				}
			}
		}
	
	}

	fclose(fr);

}

void ReadBlueBoneSet()
{
	readStack("stackAI.txt", P_BLUE);
	//gActiveSets.AddBlueBone(&greenVioletElement);
	//gActiveSets.AddBlueBone(&blueVioletElement);
	//gActiveSets.AddBlueBone(&greenBlueElement);
	////gActiveSets.AddBlueBone(&blueJadeElement);
	////gActiveSets.AddBlueBone(&violetBlueElement);//too strong


	//gActiveSets.AddBlueBone(&jadeRingElement);
	//gActiveSets.AddBlueBone(&jadeRingElement);
	//gActiveSets.AddBlueBone(&jadeRingElement);

	//gActiveSets.AddBlueBone(&greenBlobElement);
	//gActiveSets.AddBlueBone(&greenBlobElement);
}

void ReadRedBoneSet()
{
	readStack("stackHum.txt", P_RED);


	//gActiveSets.AddRedBone(&redLinkElement[0]);
	//gActiveSets.AddRedBone(&redLinkElement[2]);
	//gActiveSets.AddRedBone(&redLinkElement[3]);
	////gActiveSets.AddRedBone(&redLinkElement[4]);//red-magneta

	////gActiveSets.AddRedBone(&redLinkElement[1]); //orange-red, too strong


	//gActiveSets.AddRedBone(&yellowBlobElement);

	//gActiveSets.AddRedBone(&magnetaRingElement);
	//gActiveSets.AddRedBone(&magnetaRingElement);
}