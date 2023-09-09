#pragma once

#include "Random.h"

#include <memory>
#include <vector>


struct Network {
	int inputSize, outputSize;

	//= 1 if 0 hidden layers.
	int nLayers;

	std::vector<std::unique_ptr<float[]>> WGrads;

	std::vector<std::unique_ptr<float[]>> BGrads;

	// [in1, out1=in2, out2=in3, ...] so length nLayers + 1
	std::vector<int> sizes;

	std::vector<std::unique_ptr<float[]>> Ws;

	std::vector<std::unique_ptr<float[]>> Bs;


	// Layer by layer activations of the network. 
	std::unique_ptr<float[]> activations;
	

	// d(cost)/d(preSynAct).
	std::unique_ptr<float[]> delta;

	float* output;


	Network(int inputSize, int outputSize);

	Network(Network* n);

	// Should never be called
	Network() {
		__debugbreak();
	}

	// Should never be called
	Network(Network&& n) noexcept {
		__debugbreak();
	}

	// Should never be called
	Network(Network& n) {
		__debugbreak();
	}

	float forward(float* X, float* Y, bool accGrad);

	void updateParams(float lr, float regW, float regB);
};