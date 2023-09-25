#include "PCNN.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define CALL_KERNEL(grid, block)    <<< grid, block >>>

__global__
void GPU_tanhf(float* src, float* dst, int nC, int nR)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int index = col + row * nC;
	if (col < nC && row < nR) {
		dst[index] = tanhf(src[index]);
	}
}

__global__
void GPU_tanhf_prime(float* src, float* dst, int nC, int nR)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int index = col + row * nC;
	if (col < nC && row < nR) {
		float tfx = tanhf(src[index]);
		dst[index] = 1.0f - tfx * tfx;
	}
}


void PCNN::uploadToGPU(cublasHandle_t* _pHandle)
{

	pHandle = _pHandle;

	GPU_Xs.resize(L + 1);
	GPU_fXs.resize(L + 1);
	GPU_epsilons.resize(L + 1);


	for (int i = 0; i < L + 1; i++)
	{
		int s = batchSize * layerSizes[i] * sizeof(float);

		cudaMalloc(&GPU_Xs[i], s);
		cudaMemcpy(GPU_Xs[i], Xs[i], s, ::cudaMemcpyHostToDevice);

		cudaMalloc(&GPU_fXs[i], s);

		cudaMalloc(&GPU_epsilons[i], s);
		cudaMemcpy(GPU_epsilons[i], epsilons[i], s, ::cudaMemcpyHostToDevice);
	}


	GPU_thetas.resize(L + 1);
	GPU_biases.resize(L + 1);
	GPU_thetas[0] = nullptr;
	GPU_biases[L] = nullptr;


	for (int i = 1; i < L + 1; i++)
	{
		int s = layerSizes[i - 1] * layerSizes[i] * sizeof(float);

		cudaMalloc(&GPU_thetas[i], s);
		cudaMemcpy(GPU_thetas[i], thetas[i], s, ::cudaMemcpyHostToDevice);

		s = layerSizes[i - 1] * sizeof(float);
		cudaMalloc(&GPU_biases[i - 1], s);
		cudaMemcpy(GPU_biases[i - 1], biases[i - 1], s, ::cudaMemcpyHostToDevice);
	}
}


void PCNN::GPU_computeEpsilons(bool training)
{
	const size_t BLOCK_DIM = 16;
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);

	float one  = 1.0f;
	float zero = 0.0f;

	for (int l = 0; l < L; l++)
	{
		// epsilons = Xs - (biases(matrix !) + thetas*f(xs))
		// Xs[i+1] has batchSize columns and layerSizes[l+1] rows

		//fXs = f(Xs)
		dim3 dimGrid((int)ceil((double)batchSize / dimBlock.x), (int)ceil((double)layerSizes[l + 1] / dimBlock.y));
		GPU_tanhf CALL_KERNEL(dimGrid, dimBlock) (GPU_Xs[i + 1], GPU_fXs[i + 1], batchSize, layerSizes[l + 1]);

	
		// thetas*f(xs) 
		cublasSgemm(*pHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			batchSize, layerSizes[l+1], layerSizes[l],
			&one,
			GPU_fXs[l + 1], batchSize,
			GPU_thetas[l + 1], layerSizes[l],
			&zero,
			epsilons[l], batchSize);
	}

	for (int i = 0; i < L; i++)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int offset = dp * layerSizes[i + 1];
			for (int k = 0; k < layerSizes[i + 1]; k++) {
				buffer1[k] = tanhf(Xs[i + 1][offset + k]);
			}

			int matID = 0;
			offset = dp * layerSizes[i];
			for (int j = 0; j < layerSizes[i]; j++) {
#ifdef BIAS
				float mu = biases[i][j];
#else
				float mu = 0.0f;
#endif
				for (int k = 0; k < layerSizes[i + 1]; k++) {
					mu += thetas[i + 1][matID] * buffer1[k];
					matID++;
				}
				epsilons[i][offset + j] = Xs[i][offset + j] - mu;
			}
		}

	}

	//if reversedOrder epsilon L must be computed
	if (reversedOrder) {
		if (training)
		{
			for (int idp = 0; idp < batchSize * layerSizes[L]; idp++)
			{
				epsilons[L][idp] = Xs[L][idp] - muL[idp];
			}
		}
		else {
			std::fill(epsilons[L], epsilons[L] + batchSize * layerSizes[L], 0.0f);
		}
	}
}

//void PCNN::GPU_learn(float tlr, float regularization, float beta1, float beta2)
//{
//	float normalization = 1.0f / (float)batchSize; // not already multiplied by tlr to avoid numerical instabilities (small values)
//
//	for (int l = 1; l < L + 1; l++)
//	{
//
//	}
//	computeEpsilons(true); // theta updates only occur when there is some sort of ground truth available.
//	for (int l = 1; l < L + 1; l++)
//	{
//
//#ifdef BIAS
//		std::fill(biasesAccumulators[l - 1], biasesAccumulators[l - 1] + layerSizes[l - 1], 0.0f);
//#endif
//		std::fill(thetasAccumulators[l], thetasAccumulators[l] + layerSizes[l - 1] * layerSizes[l], 0.0f);
//
//		for (int dp = 0; dp < batchSize; dp++)
//		{
//			for (int k = 0; k < layerSizes[l]; k++) {
//				buffer1[k] = tanhf(Xs[l][dp * layerSizes[l] + k]);
//			}
//
//			for (int j = 0; j < layerSizes[l - 1]; j++) {
//				for (int k = 0; k < layerSizes[l]; k++) {
//					thetasAccumulators[l][j * layerSizes[l] + k] +=
//						buffer1[k] * epsilons[l - 1][dp * layerSizes[l - 1] + j];
//				}
//#ifdef BIAS
//				biasesAccumulators[l - 1][j] += epsilons[l - 1][dp * layerSizes[l - 1] + j];
//#endif
//			}
//		}
//
//		for (int j = 0; j < layerSizes[l - 1]; j++) {
//			for (int k = 0; k < layerSizes[l]; k++) {
//				thetasAccumulators[l][j * layerSizes[l] + k] *= normalization;
//#ifdef ADAM
//				mW[l][j * layerSizes[l] + k] = beta1 * mW[l][j * layerSizes[l] + k] +
//					(1.0f - beta1) * thetasAccumulators[l][j * layerSizes[l] + k];
//				float mhat = mW[l][j * layerSizes[l] + k] * b1f;
//
//				vW[l][j * layerSizes[l] + k] = beta2 * vW[l][j * layerSizes[l] + k] +
//					(1.0f - beta2) * powf(thetasAccumulators[l][j * layerSizes[l] + k], 2.0f);
//				float vhat = vW[l][j * layerSizes[l] + k] * b2f;
//
//				thetas[l][j * layerSizes[l] + k] = thetas[l][j * layerSizes[l] + k] * regularization +
//					tlr * mhat * powf(vhat + ADAM_epsilon, -.5f);
//#else
//				thetas[l][j * layerSizes[l] + k] = thetas[l][j * layerSizes[l] + k] * regularization +
//					(thetasAccumulators[l][j * layerSizes[l] + k] * tlr);
//#endif
//
//			}
//#ifdef BIAS
//#ifdef ADAM
//			mB[l - 1][j] = beta1 * mB[l - 1][j] +
//				(1.0f - beta1) * biasesAccumulators[l - 1][j];
//			float mhat = mB[l - 1][j] * b1f;
//
//			vB[l - 1][j] = beta2 * vB[l - 1][j] +
//				(1.0f - beta2) * powf(biasesAccumulators[l - 1][j], 2.0f);
//			float vhat = vB[l - 1][j] * b2f;
//
//			biases[l - 1][j] = biases[l - 1][j] * regularization +
//				tlr * mhat * powf(vhat + ADAM_epsilon, -.5f);
//#else
//			biases[l - 1][j] = biases[l - 1][j] * regularization +
//				(biasesAccumulators[l - 1][j] * tlr) * normalization;
//#endif
//#endif
//		}
//	}
//}
