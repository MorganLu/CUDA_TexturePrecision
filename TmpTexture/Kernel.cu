#include "Kernel.h"
#include <helper_cuda.h>
#include <cuda_runtime_api.h>


static texture<float4, 2, cudaReadModeElementType> texElnt;
static cudaArray	*texElntArray = NULL;
float  *pTmpElntArray = NULL;
float  *hArrResult    = NULL;
float  *dArrResult    = NULL;

extern "C" int InitTex( float *pData, int width, int height, int channel ) {
	int cn = UNIT*3;
	int fn = width*height*4;
	hArrResult     = new float  [cn];
	checkCudaErrors( cudaMalloc( (void**)&dArrResult, (cn)*sizeof(float) ) );

	pTmpElntArray  = new float [fn];
	memset(pTmpElntArray, 0, fn*sizeof(float));

	float *ptrData = pData;
	float *ptrElnt = pTmpElntArray;

	for (int i=0;i<width*height;i++) {
		for (int c=0;c<channel;c++, ptrData++) {
			(*(ptrElnt+c)) = (float) (*(ptrData));
		}
		ptrElnt += 4;
	}


	cudaChannelFormatDesc channelDesc; 
	channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors( cudaMallocArray(&texElntArray, &channelDesc, width, height) );
	checkCudaErrors( cudaMemcpy2DToArray(texElntArray, 0, 0, pTmpElntArray, width*sizeof(float4), width*sizeof(float4), height, cudaMemcpyHostToDevice) );

	texElnt.addressMode[0] = cudaAddressModeClamp;
	texElnt.addressMode[1] = cudaAddressModeClamp;
	texElnt.filterMode = cudaFilterModeLinear;
	texElnt.normalized = false;

	checkCudaErrors( cudaUnbindTexture(texElnt) );
	checkCudaErrors( cudaBindTextureToArray(texElnt, texElntArray, channelDesc) );
	return 0;
}

static __global__ void kernel_texElnt(float* pdata, int w, int h, int c, float stride) {
	const int gx = blockIdx.x*blockDim.x + threadIdx.x;
	const int gy = blockIdx.y*blockDim.y + threadIdx.y;
	const int gw = gridDim.x * blockDim.x;
	const int gid = gy*gw + gx;
	float2 pnt;
	pnt.x = (gx)*(stride);
	pnt.y = 0.0625f;

	float4 result = tex2D( texElnt, pnt.x + 0.5, pnt.y + 0.5f);
	pdata[gid*3 + 0] = pnt.x;
	pdata[gid*3 + 1] = pnt.y;
	pdata[gid*3 + 2] = result.x;

}

extern "C" int RunKernel( int w, int h, int c, float nBase) {
	float stride = 1.0f / nBase;
	kernel_texElnt<<< 1, UNIT >>> (dArrResult, w, h, c, stride);
	checkCudaErrors( cudaMemcpy(hArrResult, dArrResult, UNIT*3*sizeof(float), cudaMemcpyDeviceToHost) );
	return 0;
}

extern "C" int UnInitTex() {
	delete hArrResult;
	delete pTmpElntArray;
	checkCudaErrors( cudaFree(dArrResult) );
	checkCudaErrors( cudaFreeArray(texElntArray) );
	return 0;
}