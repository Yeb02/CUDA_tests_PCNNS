#pragma once


#include "PCNN.h"
#include "Network.h"
#include "MNIST.h"

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
	int lS[nL] = { labelS, 10, datapointS };

	const bool reversedOrder = false;

	if (reversedOrder) 
	{
		int temp_lS[nL] = {0};
		for (int i = 0; i < nL; i++) temp_lS[i] = lS[i];
		for (int i = 0; i < nL; i++) lS[nL-1-i] = temp_lS[i];
	}
	
	Network mlp(datapointS, labelS);


	const int batchSize = 1;
	PCNN nn(nL, lS, batchSize, reversedOrder); // TODO regularize

	float xlr = .015f / (float) batchSize;
	float tlr = .0005f / (float) batchSize;
	float regularization = 1.0f - .000f;

	ParamsDump pd;
	pd.copyPCNNsThetas(nn);

	int nInferenceSteps = 70;
	int nEpochs = 100;

	const int nTrainSamples = 60000, nTestSamples = 3000, mlpBatchSize = 10;
	for (int e = 0 ; e < nEpochs; e++)
	{

		float avgL = 0.0f;
		for (int sid = 0; sid < nTrainSamples; sid++) { 

			nn.initXs(trainDatapoints[sid], trainLabels[sid]);

			for (int i = 0; i < nInferenceSteps; i++) {
				if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, true);
				else nn.infer_Simultaneous_DataInXL(xlr, true);
			}
			if (reversedOrder) nn.learn_Simultaneous_DataInX0(tlr, regularization);
			else nn.learn_Simultaneous_DataInXL(tlr, regularization);

			/*avgL += mlp.forward(trainDatapoints[sid], trainLabels[sid], true);
			if ((sid-1)% mlpBatchSize == 0) mlp.updateParams(.1f/(float)mlpBatchSize, .000f, .0f);*/
		}

		int nCorrectAnswers = 0;
		for (int sid = 0; sid < nTestSamples; sid++)
		{
			nn.initXs(testDatapoints[sid], nullptr);

			for (int i = 0; i < nInferenceSteps; i++) {
				if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, false);
				else nn.infer_Simultaneous_DataInXL(xlr, false);
			}

			for (int i = 0; i < 10; i++)
			{
				avgL += powf(nn.output[i] - testLabels[sid][i], 2.0f);
			}
			nCorrectAnswers += isCorrectAnswer(nn.output, testLabels[sid]);


			/*mlp.forward(testDatapoints[sid], testLabels[sid], false);
			nCorrectAnswers += isCorrectAnswer(mlp.output, testLabels[sid]);*/
		}

		std::cout << "Epoch " << e << " , train loss " << avgL / (float)nTrainSamples
			<< " , test accuracy " << (float)nCorrectAnswers / (float)nTestSamples << std::endl;
	}


	int sampleID = INT_0X(10000);
	nn.initXs(testDatapoints[sampleID], testLabels[sampleID]);
	pd.copyPCNNsXs(nn);


	
	if (reversedOrder) {

		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << nn.computeEnergy() << std::endl;
			nn.infer_Simultaneous_DataInX0(xlr, true);
		}

	}
	else {

		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << nn.computeEnergy() << std::endl;
			nn.infer_Forward_DataInXL(xlr, true);
		}

		std::cout << "\n\n" << std::endl;

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn);
		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << nn.computeEnergy() << std::endl;
			nn.infer_Simultaneous_DataInXL(xlr, true);
		}
	}

	

	return 0;
}
