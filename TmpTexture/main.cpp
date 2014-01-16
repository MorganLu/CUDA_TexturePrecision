#include "Kernel.h"
#pragma warning( disable: 4996 )
float Data[4] = { 38.0f, 39.0f, 118.0f, 13.0f };


//	wrap float to fixed-point format (9.8) http://en.wikipedia.org/wiki/Fixed-point_arithmetic
inline float wrapFlt2FixPnt( float x ) {		// Fraction Ref: http://tinyurl.com/lnvyr2b
	float frac, tmp = ( x == 1.0f ? x : x - (float)(int)(x) );
	float frac256 = (float)(int)( tmp*256.0f + 0.5f );	// 0.5f for round-off
	frac = frac256 / 256.0f;
	return frac;
}

inline float EmuTex2D( float alpha, float beta, float v1, float v2, float v3, float v4 ) {
	const float AB		= wrapFlt2FixPnt(				alpha		*				beta	);
	const float AB_		= wrapFlt2FixPnt(				alpha		*	(1.0f-beta) );
	const float A_B		= wrapFlt2FixPnt( (1.0f-alpha)	*				beta	);
	const float A_B_	= wrapFlt2FixPnt( (1.0f-alpha)	* (1.0f-beta) );
	float result = v1*A_B_ + v2*AB_ + v3*A_B + v4*AB;
	return result;
}

void GoldenUnit(float nBase) {
	float stride = 1.0f / nBase;
	float curPnt[2];

	for (int x=0;x<UNIT;x++) {
		curPnt[0]    = x*stride;
		curPnt[1]    = 0.0625f;
		float alpha = ( wrapFlt2FixPnt( curPnt[0] ) );
		float beta  = ( wrapFlt2FixPnt( curPnt[1] ) );
		float result = EmuTex2D(alpha, beta, Data[0], Data[1], Data[2], Data[3]);
		hArrResult[x*3 + 0] = curPnt[0];
		hArrResult[x*3 + 1] = curPnt[1];
		hArrResult[x*3 + 2] = result;
	}
}

int main(int argc, char** argv) {
	float nBase = 32.0f;
	// GPU
	InitTex(Data, 2, 2, 1);
	RunKernel(2, 2, 1, nBase);
	freopen("D:\\GPU.txt", "w", stdout);
	for (int i=0;i<UNIT;i++) {
		float *pTmp = hArrResult + i*3;
		printf("%.010f \t %.010f \t %.010f\n", *pTmp, *(pTmp+1), *(pTmp+2) );
	}
	freopen("CON", "w", stdout);
	

	// CPU
	GoldenUnit(nBase);
	freopen("D:\\CPU.txt", "w", stdout);
	for (int i=0;i<UNIT;i++) {
		float *pTmp = hArrResult + i*3;
		printf("%.010f \t %.010f \t %.010f\n", *pTmp, *(pTmp+1), *(pTmp+2) );
	}
	freopen("CON", "w", stdout);	
	UnInitTex();
	return 0;
}