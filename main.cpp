#pragma once


#include "PCNN.h"
#include "MNIST.h"

int main() {
	const int datapointS = 28 * 28;
	const int labelS = 10;

	float** testLabels = read_mnist_labels("MNIST\\t10k-labels.idx1-ubyte", 10000);
	float** testDatapoints = read_mnist_images("MNIST\\t10k-images.idx3-ubyte", 10000);
	
	float** trainLabels = read_mnist_labels("MNIST\\train-labels.idx1-ubyte", 60000);
	float** trainDatapoints = read_mnist_images("MNIST\\train-images.idx3-ubyte", 60000);

	const int nL = 4;
	int lS[nL] = { labelS, 10, 5, datapointS };

	bool reversedOrder = true;

	if (reversedOrder) 
	{
		float temp_lS[nL] = {.0f};
		for (int i = 0; i < nL; i++) temp_lS[i] = lS[i];
		for (int i = 0; i < nL; i++) lS[nL-1-i] = temp_lS[i];
	}
	

	const int batchSize = 1;
	PCNN nn(nL, lS, batchSize, reversedOrder);

	float xlr = .01f / (float) batchSize;
	float tlr = .01f / (float)batchSize;

	ParamsDump pd;
	pd.copyPCNNsThetas(nn);

	int nSteps = 10;

	int sampleID = INT_0X(10000);
	nn.initXs(testDatapoints[sampleID], testLabels[sampleID]);
	pd.copyPCNNsXs(nn);


	for (int i = 0; i < nSteps; i++) {
		std::cout << nn.computeEnergy() << std::endl;
		nn.infer_Simultaneous_DataInX0(xlr, true);
	}

	/*for (int i = 0; i < nSteps; i++) {
		std::cout << nn.computeEnergy() << std::endl;
		nn.infer_Forward_DataInXL(xlr, true);
	}

	std::cout << "\n\n" << std::endl;

	pd.setPCNNsThetas(nn);
	pd.setPCNNsXs(nn);
	for (int i = 0; i < nSteps; i++) {
		std::cout << nn.computeEnergy() << std::endl;
		nn.infer_Simultaneous_DataInXL(xlr, true);
	}*/

	return 0;
}
