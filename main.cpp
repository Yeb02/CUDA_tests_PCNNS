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

void uploadTrainMNISTToGPU(float* GPU_trainDatapoints, float* GPU_trainLabels, float** CPU_trainDatapoints, float** CPU_trainLabels)
{
	cudaMalloc(&GPU_trainDatapoints, 784*60000);
	for (int i = 0; i < 60000; i++) {
		cudaMemcpy(GPU_trainDatapoints + 784*i, CPU_trainDatapoints[i], 784*sizeof(float), ::cudaMemcpyHostToDevice);
	}
	
	cudaMalloc(&GPU_trainLabels, 10*60000);
	for (int i = 0; i < 60000; i++) {
		cudaMemcpy(GPU_trainLabels + 10 * i, CPU_trainLabels[i], 10 * sizeof(float), ::cudaMemcpyHostToDevice);
	}
}


int main() {

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	const int datapointS = 28 * 28;
	const int labelS = 10;

	float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", 10000);
	float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", 10000);
	
	float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", 60000);
	float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", 60000);

	float* GPU_trainDatapoints;
	float* GPU_trainLabels;

	cudaMalloc(&GPU_trainDatapoints, 784 * 60000 * sizeof(float));
	for (int i = 0; i < 60000; i++) {
		cudaMemcpy(GPU_trainDatapoints + 784 * i, trainDatapoints[i], 784 * sizeof(float), ::cudaMemcpyHostToDevice);
	}
	cudaMalloc(&GPU_trainLabels, 10 * 60000 * sizeof(float));
	for (int i = 0; i < 60000; i++) {
		cudaMemcpy(GPU_trainLabels + 10 * i, trainLabels[i], 10 * sizeof(float), ::cudaMemcpyHostToDevice);
	}

	const int L = 2; // indice of the last layer. Since there is a layer 0, there is a total of L+1 layer
	int layerSizes[L+1] = { labelS, 20, datapointS };

	const bool reversedOrder = true;

	if (reversedOrder) 
	{
		int temp_lS[L + 1] = {0};
		for (int i = 0; i < L + 1; i++) temp_lS[i] = layerSizes[i];
		for (int i = 0; i < L + 1; i++) layerSizes[L-i] = temp_lS[i];
	}


	const bool corruptedInput = true;
	int corruptedIndices[datapointS]; // 0 where the input is corrupted, 1 otherwise.
	float intactFraction = 1.0f / 2.0f;
	int nCorruptedInputs = 0;
	for (int i = 0; i < datapointS; i++) {
		corruptedIndices[i] = UNIFORM_01 < intactFraction ? 1 : 0;
		nCorruptedInputs += (1 - corruptedIndices[i]);
	}

	int batchSize = 1000;
	const bool iPC = true;

	PCNN nn(L, layerSizes, batchSize, reversedOrder, corruptedInput); // TODO regularize
	ParamsDump pd;
	pd.copyPCNNsThetas(nn);

	// at batch size 1:
	// .1,  .005, 1-0    for normal   order. 20 steps.
	// at batch size 10:
	// .05, .005, 1-001. for reversed order. 100 steps.
	const float xlr = .5f;
	const float tlr = .003f;  // divided by batchSize by the network 
	const float regularization = 1.0f - .0001f; 
	//const float regularization = 1.0f - tlr * 0.05f; 

#ifdef ADAM
	const float beta1 = .9f;
	const float beta2 = .99f;
#else
	const float beta1 = .0f;
	const float beta2 = .0f;
#endif

	const int nInferenceSteps = 100;
	const int nEpochs = 10000;
	const int nTrainSamples = batchSize, nTestSamples = 10000;

	if (true) {  // full batch iPC
		const float inference_xlr = .1f;


		auto [batchedPoints, batchedLabels] = create_batches(trainDatapoints, trainLabels, nTrainSamples, batchSize);
		int nBatches = nTrainSamples / batchSize;

		nn.initXs(batchedPoints[0], batchedLabels[0]);

		PCNN testNN(L, layerSizes, 1, reversedOrder, corruptedInput);

		for (int e = 0; e < nEpochs; e++)
		{
			//if (reversedOrder) nn.iPC_Interleaved_Forward_DataInX0(tlr, xlr, regularization);
			//else nn.iPC_Interleaved_Forward_DataInXL(tlr, xlr, regularization);

			if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, true);
			else nn.infer_Simultaneous_DataInXL(xlr, true);
			nn.learn(tlr, regularization, beta1, beta2);

			if (e % 1 != 0) continue;


			std::cout << "Epoch " << e << " , energy " << std::setprecision(5) << nn.computeEnergy(true) << std::endl;


			if ((e+1) % 1000 != 0) continue;

			pd.copyPCNNsThetas(nn);
			pd.setPCNNsThetas(testNN);
#ifdef BIAS
			pd.copyPCNNsBiases(nn);
			pd.setPCNNsBiases(testNN);
#endif
			float avgL = 0.0f;
			float reconstructionError = 0.0f;
			int nCorrectAnswers = 0;
			for (int sid = 0; sid < nTestSamples; sid++)
			{
				testNN.initXs(testDatapoints[sid], nullptr, corruptedIndices);

				for (int i = 0; i < nInferenceSteps; i++) {
					if (reversedOrder) testNN.infer_Simultaneous_DataInX0(inference_xlr, false, corruptedIndices);
					else testNN.infer_Simultaneous_DataInXL(inference_xlr, false);
				}

				for (int i = 0; i < labelS; i++)
				{
					avgL += powf(testNN.output[i] - testLabels[sid][i], 2.0f);
				}
				nCorrectAnswers += isCorrectAnswer(testNN.output, testLabels[sid]);

				if (corruptedInput) {
					for (int i = 0; i < datapointS; i++)
					{
						// no need to skip uncorrupted indices, as their error will evaluate to 0.
						reconstructionError += powf(testNN.input[i] - testDatapoints[sid][i], 2.0f);
					}
				}

			}

			std::cout << "Epoch " << e << " , test loss " << std::setprecision(5) << avgL / (float)nTestSamples
				<< " , test accuracy " << std::setprecision(4) << (float)nCorrectAnswers / (float)nTestSamples;
			if (corruptedInput)
				std::cout << " , average reconstruction error " << reconstructionError / (float)(nCorruptedInputs * nTestSamples);
			std::cout << std::endl;
		}

		{
			nn.batchSize = 1;
			float avgL = 0.0f;
			float reconstructionError = 0.0f;
			int nCorrectAnswers = 0;
			for (int sid = 0; sid < nTestSamples; sid++)
			{
				nn.initXs(testDatapoints[sid], nullptr, corruptedIndices);

				for (int i = 0; i < nInferenceSteps; i++) {
					if (reversedOrder) nn.infer_Simultaneous_DataInX0(inference_xlr, false, corruptedIndices);
					else nn.infer_Simultaneous_DataInXL(inference_xlr, false);
				}

				for (int i = 0; i < labelS; i++)
				{
					avgL += powf(nn.output[i] - testLabels[sid][i], 2.0f);
				}
				nCorrectAnswers += isCorrectAnswer(nn.output, testLabels[sid]);

				if (corruptedInput) {
					for (int i = 0; i < datapointS; i++)
					{
						// no need to skip uncorrupted indices, as their error will evaluate to 0.
						reconstructionError += powf(nn.input[i] - testDatapoints[sid][i], 2.0f);
					}
				}
			}
			std::cout << "Test loss " << std::setprecision(5) << avgL / (float)nTestSamples
				<< " , test accuracy " << std::setprecision(4) << (float)nCorrectAnswers / (float)nTestSamples;
			if (corruptedInput)
				std::cout << " , average reconstruction error " << reconstructionError / (float)(nCorruptedInputs * nTestSamples);
			std::cout << std::endl;
		}

		
	}


	for (int e = 0 ; e < nEpochs; e++)
	{

		auto [batchedPoints, batchedLabels] = create_batches(trainDatapoints, trainLabels, nTrainSamples, batchSize);
		int nBatches = nTrainSamples / batchSize;

		
		nn.batchSize = batchSize;
		for (int sid = 0; sid < nBatches; sid++) {

			nn.initXs(batchedPoints[sid], batchedLabels[sid]);

			for (int i = 0; i < nInferenceSteps; i++) {
				if (reversedOrder) nn.infer_Simultaneous_DataInX0(xlr, true);
				else nn.infer_Simultaneous_DataInXL(xlr, true);

				if (iPC) nn.learn(tlr, regularization, beta1, beta2);
			}
			if (!iPC) nn.learn(tlr, regularization, beta1, beta2);

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

			for (int i = 0; i < labelS; i++)
			{
				avgL += powf(nn.output[i] - testLabels[sid][i], 2.0f);
			}
			nCorrectAnswers += isCorrectAnswer(nn.output, testLabels[sid]);

			if (corruptedInput) {
				for (int i = 0; i < datapointS; i++)
				{
					// no need to skip uncorrupted indices, as their error will evaluate to 0.
					reconstructionError += powf(nn.input[i] - testDatapoints[sid][i], 2.0f);
				}
			}

		}

		std::cout << "Epoch " << e << " , test loss " << std::setprecision(5) << avgL / (float)nTestSamples
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
	

	const bool training = false;

	auto [batchedPoints, batchedLabels] = create_batches(testDatapoints, testLabels, 10000, batchSize);
	int nBatches = nTestSamples / batchSize;

	int sampleID = INT_0X(nBatches);
	std::cout << sampleID << "\n" << std::endl;
	if (training) nn.initXs(batchedPoints[sampleID], batchedLabels[sampleID]);
	else nn.initXs(batchedPoints[sampleID], nullptr);
	pd.copyPCNNsXs(nn, batchSize);


	if (reversedOrder) { 
		/* layerSizes = { labelS, 10, 10, 10, datapointS }, xlr = .25, tlr = .0005f/10.f
		 preliminary results, a>b means a better than b. 
		  
		 1 - on the speed of energy minimization (compared after 10 steps)
		 supervised   (training=true)  : F << B << S
		 unsupervised (training=false) : F ~= B ~= S
		 iPC (batchSize=10, supervised): F ~= B ~= S 
		 2 - on the minimum of the energy at convergence (compared after 200 steps)
		 supervised   (training=true)  : F  S  B 
		 unsupervised (training=false) : F  S  B
		 iPC (batchSize=10, supervised): F  S  B*/

		std::cout << "Initial energy: " << nn.computeEnergy(training) << "\n\n" << std::endl;
		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << "  F, total " << nn.computeEnergy(training) << " , ";
			for (int l = 0; l < L+(training?1:0); l++) {
				std::cout << nn.computeEpsilonNorm(l) << " ";
			}
			std::cout << std::endl;

			nn.infer_Forward_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  F, total " << nn.computeEnergy(training) << " , ";
		for (int l = 0; l < L + (training ? 1 : 0); l++) {
			std::cout << nn.computeEpsilonNorm(l) << " ";
		}
		std::cout << "\n" << std::endl;

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		//std::cout << "  B " << nn.computeEnergy(training) << std::endl;
		for (int i = 0; i < nInferenceSteps; i++) {
			nn.infer_Backward_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  B, total " << nn.computeEnergy(training) << " , ";
		for (int l = 0; l < L + (training ? 1 : 0); l++) {
			std::cout << nn.computeEpsilonNorm(l) << " ";
		}
		std::cout << "\n" << std::endl;
		

		pd.setPCNNsThetas(nn);
		pd.setPCNNsXs(nn, batchSize);
		for (int i = 0; i < nInferenceSteps; i++) {
			std::cout << "  S, total " << nn.computeEnergy(training) << " , ";
			for (int l = 0; l < L + (training ? 1 : 0); l++) {
				std::cout << nn.computeEpsilonNorm(l) << " ";
			}
			std::cout << std::endl;
			nn.infer_Simultaneous_DataInX0(xlr, training, corruptedIndices);
			if (iPC) nn.learn(tlr, regularization);
		}
		std::cout << "  S, total " << nn.computeEnergy(training) << " , ";
		for (int l = 0; l < L + (training ? 1 : 0); l++) {
			std::cout << nn.computeEpsilonNorm(l) << " ";
		}
		std::cout << std::endl;
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
