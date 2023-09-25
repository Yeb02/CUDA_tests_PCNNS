#pragma once

#include <memory>
#include <vector>
#include <iostream>

#include "Random.h"
#include "Config.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#pragma comment(lib, "cublas.lib") // Not linked by default in the cuda template. 

class PCNN
{
	friend class ParamsDump;

private:
	int L; // layer 0, 1, ... layer L. (so L+1 layers in total, and L linear transformations)
	int* layerSizes; // Size L+1

	std::vector<float*> Xs; // Size L+1
	std::vector<float*> epsilons; // Size L+1

	std::vector<float*> thetas; // Size L+1, but the first element is nullptr
	std::vector<float*> thetasAccumulators; // Size L+1, but the first element is nullptr

#ifdef BIAS
	std::vector<float*> biases; // Size L+1, but the last  element is nullptr
	std::vector<float*> biasesAccumulators; // Size L+1, but the last  element is nullptr
#endif

#ifdef ADAM
	std::vector<float*> mW; // Size L+1, but the first element is nullptr
	std::vector<float*> vW; // Size L+1, but the first element is nullptr
	std::vector<float*> mB; // Size L+1, but the first element is nullptr
	std::vector<float*> vB; // Size L+1, but the first element is nullptr
#endif


	cublasHandle_t* pHandle;

	std::vector<float*> GPU_Xs; // Size L+1
	std::vector<float*> GPU_fXs; // Size L+1. Used to store the temporary values f(X) and f'(x). Has no equivalent in CPU memory
	std::vector<float*> GPU_epsilons; // Size L+1
	std::vector<float*> GPU_thetas; // Size L+1, but the first element is nullptr
	std::vector<float*> GPU_biases; // Size L+1, but the last element is nullptr





	float* buffer1; // array to store temporary values during inference.

	
	bool reversedOrder; // true iff the datapoint is presented at x0 instead of at xL.
	bool corruptedInput; // true if at runtime (i.e. not training) the input can be incomplete or corrupted
	float* muL;    // only used when the datapoint is presented to x0 and label to xL.

	// updates all relevant epsilons
	void computeEpsilons(bool training);

	void GPU_computeEpsilons(bool training);

public:

	int batchSize; // size of the batch, if > 1 iPC should be used.

	float* output;
	float* input;

	~PCNN() 
	{
		for (int i = 0; i < L; i++) {
			delete[] Xs[i];
			delete[] epsilons[i];

			delete[] thetas[i];
			delete[] thetasAccumulators[i];


#ifdef BIAS
			delete[] biases[i]; 
			delete[] biasesAccumulators[i]; 
#endif

#ifdef ADAM
			delete[] mB[i]; 
			delete[] vB[i]; 
			delete[] mW[i]; 
			delete[] vW[i]; 
#endif
		}

		delete[] buffer1;	
		delete[] muL;	
	}


	PCNN(int _nL, int* _lS, int _datasetSize, bool _reversedOrder, bool _corruptedInput = false);


	void initXs(float* datapoint, float* label=nullptr, int* corruptedIndices = nullptr);


	// transfers all necessary parameters to GPU. Must be called before using any "GPU_" function.
	void uploadToGPU(cublasHandle_t* _pHandle);

	void GPU_learn(float tlr, float regularization, float beta1 = .0f, float beta2 = .0f);

	void GPU_Infer_Simultaneous_DataInXL(float xlr, bool training);



	// for testing purposes only
	float computeEnergy(bool training);

	// computes sum of squared epsilons at a given layer (averaged of samples in the batch)
	float computeEpsilonNorm(int layer) {
		float s = 0.0f;
		for (int idp = 0; idp < layerSizes[layer] * batchSize; idp++) {
			s += powf(epsilons[layer][idp], 2.0f);
		}
		return s / (float)batchSize;
	}



	// updates thetas, whatever end the input is at and whatever
	// the direction of propagation.
	void learn(float tlr, float regularization, float beta1=.0f, float beta2=.0f);



	// interleaves x updates and theta updates. Only called during training, when x0 is fixed at the label
	void iPC_Interleaved_Forward_DataInXL(float tlr, float xlr, float regularization, float beta1 = .0f, float beta2 = .0f);

	// updates all Xs simultaneously, when xL is the datapoint 
	void infer_Simultaneous_DataInXL(float xlr, bool training);

	// updates Xs in a forward fashion, when xL is the datapoint 
	void infer_Forward_DataInXL(float xlr, bool training);

	// updates Xs in a backward fashion, when xL is the datapoint 
	void infer_Backward_DataInXL(float xlr, bool training);



	// interleaves x updates and theta updates. Only called during training, when xL is fixed at the label
	void iPC_Interleaved_Forward_DataInX0(float tlr, float xlr, float regularization, float beta1 = .0f, float beta2 = .0f);

	// updates all Xs simultaneously, when x0 is the datapoint 
	void infer_Simultaneous_DataInX0(float xlr, bool training, int* corruptedIndices = nullptr);

	// updates Xs in a forward fashion, when x0 is the datapoint 
	void infer_Forward_DataInX0(float xlr, bool training, int* corruptedIndices = nullptr);

	// updates Xs in a backward fashion, when x0 is the datapoint 
	void infer_Backward_DataInX0(float xlr, bool training, int* corruptedIndices = nullptr);
};


// util to store the parameters of a PCNN, and its activations.
class ParamsDump
{
	std::vector<float*> thetas;
	std::vector<float*> Xs;
#ifdef BIAS
	std::vector<float*> biases;
#endif

public:
	ParamsDump() {};

	~ParamsDump() {
		for (int i = 0; i < thetas.size(); i++) {
			delete[] thetas[i];
			delete[] Xs[i];
#ifdef BIAS
			delete[] biases[i];
#endif
		}
	}

	void copyPCNNsThetas(PCNN& nn) {
		thetas.resize(nn.L+1);
		thetas[0] = nullptr;
		for (int i = 1; i < nn.L+1; i++) {
			int s = nn.layerSizes[i - 1] * nn.layerSizes[i];
			thetas[i] = new float[s];
			std::copy(nn.thetas[i], nn.thetas[i] + s, thetas[i]);
		}
	}

#ifdef BIAS
	void copyPCNNsBiases(PCNN& nn) {
		biases.resize(nn.L + 1);
		biases[nn.L] = nullptr;
		for (int i = 0; i < nn.L; i++) {
			int s = nn.layerSizes[i];
			biases[i] = new float[s];
			std::copy(nn.biases[i], nn.biases[i] + s, biases[i]);
		}
	}
#endif

	void copyPCNNsXs(PCNN& nn, int batchSize) {
		Xs.resize(nn.L + 1);
		for (int i = 0; i < nn.L + 1; i++) {
			int s = nn.layerSizes[i] * batchSize;
			Xs[i] = new float[s];
			std::copy(nn.Xs[i], nn.Xs[i] + s, Xs[i]);
		}
	}

	void setPCNNsThetas(PCNN& nn) {
		for (int i = 1; i < nn.L + 1; i++) {
			int s = nn.layerSizes[i - 1] * nn.layerSizes[i];
			std::copy(thetas[i], thetas[i] + s, nn.thetas[i]);
		}
	}

	void setPCNNsXs(PCNN& nn, int batchSize) {
		for (int i = 0; i < nn.L + 1; i++) {
			int s = nn.layerSizes[i] * batchSize;
			std::copy(Xs[i], Xs[i] + s, nn.Xs[i]);
		}
	}

#ifdef BIAS
	void setPCNNsBiases(PCNN& nn) {
		for (int i = 0; i < nn.L; i++) {
			int s = nn.layerSizes[i];
			std::copy(biases[i], biases[i] + s, nn.biases[i]);
		}
	}
#endif
};

