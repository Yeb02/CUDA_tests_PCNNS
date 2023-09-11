#pragma once

#include <memory>
#include <vector>
#include <iostream>

#include "Random.h"


class PCNN
{
	friend class ParamsDump;

private:
	int L; // layer 0, 1, ... layer L. (so L+1 layers in total, and L linear transformations)
	int batchSize; // size of the batch, if > 1 iPC should be used.
	int* layerSizes; // Size L+1

	std::vector<float*> Xs; // Size L+1
	std::vector<float*> epsilons; // Size L+1
	std::vector<float*> thetas; // Size L+1, but the first element is nullptr

	float* buffer1; // array to store temporary values during inference.
	float* buffer2; // array to store temporary values during inference.

	
	bool reversedOrder; // true iff the datapoint is presented at x0 instead of at xL.
	bool corruptedInput; // true if at runtime (i.e. not training) the input can be incomplete or corrupted
	float* muL;    // only used when the datapoint is presented to x0 and label to xL.

	// is used before Xs update and between Xs update and thetas update. 

	void computeEpsilons();

public:

	float* output;

	~PCNN() 
	{
		for (int i = 0; i < L; i++) {
			delete[] Xs[i];
			delete[] epsilons[i];
			delete[] thetas[i];
		}

		delete[] buffer1;	
		delete[] buffer2;	
		delete[] muL;	
	}

	PCNN(int _nL, int* _lS, int _datasetSize, bool _reversedOrder, bool _corruptedInput = false);

	void initXs(float* datapoint, float* label=nullptr);

	// for testing purposes only
	float computeEnergy();




	// updates all Xs simultaneously, when xL is the datapoint 
	void infer_Simultaneous_DataInXL(float xlr, bool training);

	// updates Xs in a forward fashion, when xL is the datapoint 
	void infer_Forward_DataInXL(float xlr, bool training);

	// updates thetas in a forward fashion, when xL is the datapoint 
	void learn_DataInXL(float tlr, float regularization);



	// updates all Xs simultaneously, when x0 is the datapoint 
	void infer_Simultaneous_DataInX0(float xlr, bool training);

	// updates Xs in a forward fashion, when x0 is the datapoint 
	void infer_Forward_DataInX0(float xlr, bool training);

	// updates all thetas simultaneously, when x0 is the datapoint 
	void learn_DataInX0(float tlr, float regularization);
};


// util to store the parameters of a PCNN
class ParamsDump
{
	std::vector<float*> thetas;
	std::vector<float*> Xs;

public:
	ParamsDump() {};

	~ParamsDump() {
		for (int i = 0; i < thetas.size(); i++) {
			delete[] thetas[i];
			delete[] Xs[i];
		}
	}

	void copyPCNNsThetas(PCNN& nn) {
		thetas.resize(nn.L);
		thetas[0] = nullptr;
		for (int i = 1; i < nn.L; i++) {
			int s = nn.layerSizes[i - 1] * nn.layerSizes[i];
			thetas[i] = new float[s];
			std::copy(nn.thetas[i], nn.thetas[i] + s, thetas[i]);
		}
	}

	void copyPCNNsXs(PCNN& nn) {
		Xs.resize(nn.L);
		for (int i = 0; i < nn.L; i++) {
			int s = nn.layerSizes[i];
			Xs[i] = new float[s];
			std::copy(nn.Xs[i], nn.Xs[i] + s, Xs[i]);
		}
	}

	void setPCNNsThetas(PCNN& nn) {
		for (int i = 1; i < nn.L; i++) {
			int s = nn.layerSizes[i - 1] * nn.layerSizes[i];
			std::copy(thetas[i], thetas[i] + s, nn.thetas[i]);
		}
	}

	void setPCNNsXs(PCNN& nn) {
		for (int i = 0; i < nn.L; i++) {
			int s = nn.layerSizes[i];
			std::copy(Xs[i], Xs[i] + s, nn.Xs[i]);
		}
	}
};

