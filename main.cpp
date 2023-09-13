#pragma once

#ifdef _DEBUG
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);
#endif


#include "PCNN.h"
#include "Network.h"
#include "MNIST.h"

#include <iomanip>

int isCorrectAnswer(float* out, float* y) {
	float outMax = -1000.0f;
	int imax = -1;
	for (int i = 0; i < 10; i++) 
	{
		if (outMax < out[i]) {
			outMax = out[i];
			imax = i;
		}
	}

	return y[imax] == 1.0f ? 1 : 0;
}


int main() {
	const int datapointS = 28 * 28;
	const int labelS = 10;

	float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", 10000);
	float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", 10000);
	
	float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", 60000);
	float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", 60000);

	const int nL = 3;
	int layerSizes[nL] = { labelS, 10, datapointS };

	const bool reversedOrder = true;

	if (reversedOrder) 
	{
		int temp_lS[nL] = {0};
		for (int i = 0; i < nL; i++) temp_lS[i] = layerSizes[i];
		for (int i = 0; i < nL; i++) layerSizes[nL-1-i] = temp_lS[i];
	}

	const int batchSize = 1; 

	// .1, .005 for normal order
	const float xlr = .02f; 
	const float tlr = .01f / (float)batchSize; // i have noticed that if x has not converged (iPC, ...),
	const float regularization = 1.0f;// -.015f; // ,tlr must be tiny, otherwise thetas explode.

	const bool corruptedInput = false;
	int corruptedIndices[784]; // 0 where the input is corrupted, 1 otherwise.
	float intactFraction = 1.0f / 2.0f;
	int nCorruptedInputs = 0;
	for (int i = 0; i < 784; i++) {
		corruptedIndices[i] = UNIFORM_01 < intactFraction ? 1 : 0;
		nCorruptedInputs += (1 - corruptedIndices[i]);
	}

	PCNN nn(nL-1, layerSizes, batchSize, reversedOrder, corruptedInput); // TODO regularize
	ParamsDump pd;
	pd.copyPCNNsThetas(nn);


	const int nInferenceSteps = 20;
	const int nEpochs = 1000;
	const int nTrainSamples = 3000, nTestSamples = 600;
	for (int e = 0 ; e < nEpochs; e++)
	{

		auto [batchedPoints, batchedLabels] = create_batches(trainDatapoints, trainLabels, nTrainSamples, batchSize);
		int nBatches = nTestSamples / batchSize;

		
		nn.batchSize = batchSize;
		for (int sid = 0; sid < nBatches; sid++) {

			nn.initXs(batchedPoints[sid], batchedLabels[sid]);

			for (int i = 0; i < nInferenceSteps; i++) {
				if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, true);
				else nn.infer_Simultaneous_DataInXL(xlr, true);
			}
			nn.learn(tlr, regularization);

		}

		float avgL = 0.0f;
		float reconstructionError = 0.0f;
		int nCorrectAnswers = 0;
		nn.batchSize = 1;
		for (int sid = 0; sid < nTestSamples; sid++)
		{
			nn.initXs(testDatapoints[sid], nullptr, corruptedIndices);

			for (int i = 0; i < nInferenceSteps; i++) {
				if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, false, corruptedIndices);
				else nn.infer_Simultaneous_DataInXL(xlr, false);
			}

			for (int i = 0; i < 10; i++)
			{
				avgL += powf(nn.output[i] - testLabels[sid][i], 2.0f);
			}
			nCorrectAnswers += isCorrectAnswer(nn.output, testLabels[sid]);

			for (int i = 0; i < 784; i++)
			{
				// no need to skip uncorrupted indices, as their error will evaluate to 0.
				reconstructionError += powf(nn.input[i] - testDatapoints[sid][i], 2.0f);
			}

		}

		std::cout << "Epoch " << e << " , test loss " << std::setprecision(5) << avgL / (float)nTrainSamples
			<< " , test accuracy " << std::setprecision(4) << (float)nCorrectAnswers / (float)nTestSamples;
		if (corruptedInput) 
			std::cout << " , average reconstruction error " << reconstructionError / (float)(nCorruptedInputs * nTestSamples);
		std::cout << std::endl;

		for (int i = 0; i < nBatches; i++) {
			delete[] batchedPoints[i];
			delete[] batchedLabels[i];
		}
		delete[] batchedPoints;
		delete[] batchedLabels;
	}

	// testing an MLP's performance on the dataset
	if (false) {
		Network mlp(datapointS, labelS);
		const int mlpBatchSize = 10;

		for (int e = 0; e < nEpochs; e++)
		{

			float avgL = 0.0f;
			for (int sid = 0; sid < nTrainSamples; sid++) {
				avgL += mlp.forward(trainDatapoints[sid], trainLabels[sid], true);
				if ((sid-1)% mlpBatchSize == 0) mlp.updateParams(.1f/(float)mlpBatchSize, .000f, .0f);
			}

			int nCorrectAnswers = 0;
			for (int sid = 0; sid < nTestSamples; sid++)
			{
				mlp.forward(testDatapoints[sid], testLabels[sid], false);
				nCorrectAnswers += isCorrectAnswer(mlp.output, testLabels[sid]);
			}

			std::cout << "Epoch " << e << " , train loss " << avgL / (float)nTrainSamples
				<< " , test accuracy " << (float)nCorrectAnswers / (float)nTestSamples << std::endl;
		}
	}
	

	const bool training = true;
	const bool iPC = true; 

	auto [batchedPoints, batchedLabels] = create_batches(testDatapoints, testLabels, nTestSamples, batchSize);
	int nBatches = nTestSamples / batchSize;

	int sampleID = INT_0X(nBatches);
	std::cout << sampleID << "\n" << std::endl;
	if (training) nn.initXs(batchedPoints[sampleID], batchedLabels[sampleID]);
	else nn.initXs(batchedPoints[sampleID], nullptr);
	pd.copyPCNNsXs(nn, batchSize);


	if (reversedOrder) { 
		// layerSizes = { labelS, 10, 10, 10, datapointS }, xlr = .25, tlr = .0005f/10.f
		// preliminary results, a>b means a better than b. 
		//  
		// 1 - on the speed of energy minimization (compared after 10 steps)
		// supervised   (training=true)  : F << B << S
		// unsupervised (training=false) : F ~= B ~= S
		// iPC (batchSize=10, supervised): F ~= B ~= S 
		// 2 - on the minimum of the energy at convergence (compared after 200 steps)
		// supervised   (training=true)  : F  S  B 
		// unsupervised (training=false) : F  S  B
		// iPC (batchSize=10, supervised): F  S  B

		std::cout << "Initial energy: " << nn.computeEnergy(training) << "\n" << std::endl;
		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << "  F " << nn.computeEnergy(training) << std::endl;
			nn.infer_Forward_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  F " << nn.computeEnergy(training) << "\n" << std::endl;

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		//std::cout << "  B " << nn.computeEnergy(training) << std::endl;
		for (int i = 0; i < nInferenceSteps; i++) {
			nn.infer_Backward_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  B " << nn.computeEnergy(training) << "\n" << std::endl;

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		for (int i = 0; i < nInferenceSteps; i++) {
			//std::cout << "  S " << nn.computeEnergy(training) << std::endl;
			nn.infer_Simultaneous_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  S " << nn.computeEnergy(training) << std::endl;
	}
	else { 
		// layerSizes = { labelS, 10, 10, 10, datapointS }, xlr = .25, batchSize = 1
		// preliminary results, a>b means a better than b. 
		//  
		// 1 - on the speed of energy minimization (compared after 10 steps)
		// supervised   (training=true)  : F ~= B ~= S
		// unsupervised (training=false) : F ~= B ~= S
		// 2 - on the minimum of the energy at convergence (compared after 200 steps)
		// supervised   (training=true)  : F ~= B ~= S
		// unsupervised (training=false) : F ~> B ~> S

		for (int i = 0; i < nInferenceSteps; i++) {
			nn.infer_Forward_DataInXL(xlr, training);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  F " << nn.computeEnergy(training) << std::endl;
		
		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		for (int i = 0; i < nInferenceSteps; i++) {
			nn.infer_Backward_DataInXL(xlr, training);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  B " << nn.computeEnergy(training) << std::endl;

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		for (int i = 0; i < nInferenceSteps; i++) {
			nn.infer_Simultaneous_DataInXL(xlr, training);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  S " << nn.computeEnergy(training) << std::endl;
	}

	

	return 0;
}
