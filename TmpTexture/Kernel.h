#include <iostream>
#include <iomanip>

#define UNIT 32

extern float  *hArrResult;
extern "C" int InitTex( float *pData, int width, int height, int channel );
extern "C" int RunKernel( int w, int h, int c, float nBase);
extern "C" int UnInitTex();
