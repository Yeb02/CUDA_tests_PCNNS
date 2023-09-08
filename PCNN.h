#pragma once

#include <memory>
#include <vector>
#include <iostream>

#include "Random.h"


class PCNN
{
	friend class ParamsDump;

private:
	int nL, dS;
	int* lS;

	std::vector<float*> Xs;
	std::vector<float*> epsilons;
	std::vector<float*> thetas;

	float* buffer; // array to store temporary values during inference.
	
	// is used before Xs update and between Xs update and thetas update
	void computeEpsilons(); 

public:

	~PCNN() 
	{
		for (int i = 0; i < nL; i++) {
			delete[] Xs[i];
			delete[] epsilons[i];
			delete[] thetas[i];
		}

		delete[] buffer;	
	}

	PCNN(int _nL, int* _lS, int datasetSize);


	void initXs(float* x0, float* xL);

	// updates Xs and thetas
	void iPCStep(float xlr, float tlr);

	// for testing purposes only
	float computeEnergy()
	{
		float E = 0.0f;

		computeEpsilons();

		for (int l = 0; l < nL - 1; l++) {
			for (int dp = 0; dp < dS; dp++) {
				for (int i = 0; i < lS[l]; i++) {
					E += powf(epsilons[l][dp * lS[l] + i], 2.0f);
				}
			}
		}

		return E;
	}




	// updates all Xs simultaneously, when xL is the datapoint 
	void infer_Simultaneous_DataInXL(float xlr, bool training);

	// updates all thetas simultaneously, when xL is the datapoint 
	void learn_Simultaneous_DataInXL(float tlr);



	// updates Xs in a forward fashion, when xL is the datapoint 
	void infer_Forward_DataInXL(float xlr, bool training);

	// updates thetas in a forward fashion, when xL is the datapoint 
	void learn_Forward_DataInXL(float tlr);



	// updates all Xs simultaneously, when x0 is the datapoint 
	void infer_Simultaneous_DataInX0(float xlr, bool training);

	// updates all thetas simultaneously, when x0 is the datapoint 
	void learn_Simultaneous_DataInX0(float tlr);



	// updates Xs in a forward fashion, when x0 is the datapoint 
	void infer_Forward_DataInX0(float xlr, bool training);

	// updates thetas in a forward fashion, when x0 is the datapoint 
	void learn_Forward_DataInX0(float tlr);

};


// util to store the learned parameters of a PCNN
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
		thetas.resize(nn.nL);
		thetas[0] = nullptr;
		for (int i = 1; i < nn.nL; i++) {
			int s = nn.lS[i - 1] * nn.lS[i];
			thetas[i] = new float[s];
			std::copy(nn.thetas[i], nn.thetas[i] + s, thetas[i]);
		}
	}

	void copyPCNNsXs(PCNN& nn) {
		Xs.resize(nn.nL);
		for (int i = 0; i < nn.nL; i++) {
			int s = nn.lS[i];
			Xs[i] = new float[s];
			std::copy(nn.Xs[i], nn.Xs[i] + s, Xs[i]);
		}
	}

	void setPCNNsThetas(PCNN& nn) {
		for (int i = 1; i < nn.nL; i++) {
			int s = nn.lS[i - 1] * nn.lS[i];
			std::copy(thetas[i], thetas[i] + s, nn.thetas[i]);
		}
	}

	void setPCNNsXs(PCNN& nn) {
		for (int i = 0; i < nn.nL; i++) {
			int s = nn.lS[i];
			std::copy(Xs[i], Xs[i] + s, nn.Xs[i]);
		}
	}
};

