#pragma once

#include "Network.h"

Network::Network(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize)
{
	nLayers = 2; // 1 if no hidden

	sizes.push_back(inputSize);
	for (int i = 0; i < nLayers - 1; i++) { 
		sizes.push_back(30);
	}
	sizes.push_back(outputSize);


	// init Ws and Bs
	for (int i = 0; i < nLayers; i++)
	{
		float f = powf((float)sizes[i], -.5f);
		int sW = sizes[i] * sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = NORMAL_01;
			//Ws[i][j] = NORMAL_01 * f;
		}
		WGrads.emplace_back(new float[sW]);
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);


		int sB = sizes[i + 1];
		Bs.emplace_back(new float[sB]);
		for (int j = 0; j < sB; j++) {
			//Bs[i][j] = .0f;
			Bs[i][j] = NORMAL_01;
			//Bs[i][j] = NORMAL_01*.2f;
		}
		BGrads.emplace_back(new float[sB]);
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}

	int activationS = 0;
	for (int i = 0; i < nLayers + 1; i++) {
		activationS += sizes[i];
	}
	activations = std::make_unique<float[]>(activationS);
	delta = std::make_unique<float[]>(activationS - inputSize);

	output = &activations[activationS - outputSize];
}


float Network::forward(float* X, float* Y, bool accGrad)
{

	// forward
	float* prevActs = &activations[0];
	std::copy(X, X + inputSize, prevActs);
	float* currActs = &activations[inputSize];
	for (int i = 0; i < nLayers; i++) {

		int matID = 0;
		for (int j = 0; j < sizes[i + 1]; j++) {

			currActs[j] = Bs[i][j];
			for (int k = 0; k < sizes[i]; k++) {
				currActs[j] += Ws[i][matID] * prevActs[k];
				matID++;
			}
			//currActs[j] = 1.0f / (1 + expf(-currActs[j]));
			currActs[j] = tanhf(currActs[j]);
		}

		prevActs = currActs;
		currActs = currActs + sizes[i + 1];
	}

	float loss = 0.0f; 
	for (int i = 0; i < outputSize; i++) // euclidean distance loss
	{
		loss += powf(output[i] - Y[i], 2.0f);
	}

	if (!accGrad) {
		return loss;
	}


	float* prevDelta;
	float* currDelta = &delta[0];
	currActs = output;
	prevActs = output - sizes[nLayers - 1];

	for (int i = 0; i < outputSize; i++)
	{
		//currDelta[i] = (currActs[i] - Y[i]) * (1.0f - currActs[i]) * currActs[i];
		currDelta[i] = (currActs[i] - Y[i]) * (1.0f - currActs[i] * currActs[i]);
	}

	for (int i = nLayers - 1; i >= 0; i--) {

		// w
		int matID = 0;
		for (int j = 0; j < sizes[i + 1]; j++) {
			for (int k = 0; k < sizes[i]; k++) {
				WGrads[i][matID] += currDelta[j] * prevActs[k];
				matID++;
			}
		}

		// b
		for (int j = 0; j < sizes[i + 1]; j++)
		{
			BGrads[i][j] += currDelta[j];
		}

		if (i == 0) break;

		prevDelta = currDelta;
		currDelta = currDelta + sizes[i + 1];

		currActs = prevActs;
		prevActs = currActs - sizes[i - 1];

		for (int k = 0; k < sizes[i]; k++) {
			currDelta[k] = 0.0f;
		}

		for (int k = 0; k < sizes[i]; k++) {
			for (int j = 0; j < sizes[i + 1]; j++) {
				currDelta[k] += Ws[i][j * sizes[i] + k] * prevDelta[j];
			}
		}
		for (int k = 0; k < sizes[i]; k++) {
			//currDelta[k] *= (1.0f - currActs[k]) * currActs[k];
			currDelta[k] *= (1.0f - currActs[k] * currActs[k]);
		}

	}

	return loss;
}


void Network::updateParams(float lr, float regW, float regB)
{
	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = Ws[i][j] * (1.0f - regW) - WGrads[i][j] * lr;
		}
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);

		int sB = sizes[i + 1];
		for (int j = 0; j < sB; j++) {
			Bs[i][j] = Bs[i][j] * (1.0f - regB) - BGrads[i][j] * lr;
		}
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}
}


Network::Network(Network* n) : Network(n->inputSize, n->outputSize)
{
	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		std::copy(n->Ws[i].get(), n->Ws[i].get() + sW, Ws[i].get());

		int sB = sizes[i + 1];
		std::copy(n->Bs[i].get(), n->Bs[i].get() + sB, Bs[i].get());
	}
}