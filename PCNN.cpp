#include "PCNN.h"


PCNN::PCNN(int _L, int* _lS, int _dS, bool _reversedOrder, bool _corruptedInput) :
	L(_L), layerSizes(_lS), batchSize(_dS), reversedOrder(_reversedOrder), corruptedInput(_corruptedInput)
{

	Xs.resize(L+1);
	epsilons.resize(L+1);

	int lSmax = layerSizes[0];
	for (int i = 0; i < L + 1; i++)
	{
		Xs[i] = new float[batchSize * layerSizes[i]];
		epsilons[i] = new float[batchSize * layerSizes[i]];

		if (layerSizes[i] > lSmax) lSmax = layerSizes[i];
	}

	buffer1 = new float[lSmax];


	thetas.resize(L+1);
	thetasAccumulators.resize(L+1);
	
#ifdef BIAS
	biases.resize(L+1);
	biasesAccumulators.resize(L+1);
#endif

#ifdef ADAM
	mW.resize(L + 1);
	mB.resize(L + 1);
	vW.resize(L + 1);
	vB.resize(L + 1);
#endif


	thetas[0] = nullptr;
	thetasAccumulators[0] = nullptr;

#ifdef ADAM
	vW[0] = nullptr;
	mW[0] = nullptr;
	vB[L] = nullptr;
	mB[L] = nullptr;
#endif

#ifdef BIAS
	biases[L] = nullptr;
	biasesAccumulators[L] = nullptr;
#endif

	for (int i = 1; i < L+1; i++) 
	{
		int s = layerSizes[i - 1] * layerSizes[i];

		thetas[i] = new float[s];
		thetasAccumulators[i] = new float[s];

		//float f = powf((float)s, -.5f); // HE init
		//for (int j = 0; j < s; j++) thetas[i][j] = NORMAL_01 * f;

		// https://www.mrcbndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf
		float f = 8.0f * sqrtf(6.0f / (float)(layerSizes[i - 1] + layerSizes[i]));
		for (int j = 0; j < s; j++) thetas[i][j] = (UNIFORM_01 - .5f) * f;


#ifdef ADAM
		mW[i] = new float[s];
		vW[i] = new float[s];
		mB[i-1] = new float[layerSizes[i-1]];
		vB[i-1] = new float[layerSizes[i-1]];
#endif

#ifdef BIAS
		biases[i - 1] = new float[layerSizes[i - 1]];
		biasesAccumulators[i - 1] = new float[layerSizes[i - 1]];


		//std::fill(biases[i - 1], biases[i - 1] + layerSizes[i - 1], 0.0f);
		for (int j = 0; j < layerSizes[i - 1]; j++) biases[i - 1][j] = (UNIFORM_01 - .5f) * .2f;
#endif
	}


	if (reversedOrder) {
		output = Xs[L];
		input = Xs[0];
		muL = new float[batchSize * layerSizes[L]];
	}
	else {
		output = Xs[0];
		input = Xs[L];
		muL = nullptr;
	}
}


void PCNN::initXs(float* datapoint, float* label, int* corruptedIndices)
{
	

	if (reversedOrder) {
		
		if (corruptedInput && corruptedIndices!=nullptr && label == nullptr) { // to train with corrupted inputs, add   && label == nullptr 
			for (int dp = 0; dp < batchSize; dp++) {
				int offset = dp * layerSizes[0];
				for (int i = 0; i < layerSizes[0]; i++) {
					Xs[0][offset + i] = (float)corruptedIndices[i] * datapoint[offset + i];
				}
			}
		}
		else {
			std::copy(datapoint, datapoint + layerSizes[0] * batchSize, Xs[0]);
		}

		if (label == nullptr)
		{
			std::fill(muL, muL + layerSizes[L] * batchSize, 0.0f);

			//std::fill(Xs[L], Xs[L] + layerSizes[L] * batchSize, 0.0f);
			float f = powf((float)layerSizes[L], -.5f);
			for (int j = 0; j < batchSize * layerSizes[L]; j++) Xs[L][j] = NORMAL_01 * f;
		}
		else {
			std::copy(label, label + layerSizes[L] * batchSize, muL);

			std::copy(label, label + batchSize * layerSizes[L], Xs[L]);
			//std::fill(Xs[L], Xs[L] + layerSizes[L] * batchSize, 0.0f);
			/*float f = powf((float)layerSizes[L], -.5f);
			for (int j = 0; j < batchSize * layerSizes[L]; j++) Xs[L][j] = NORMAL_01 * f;*/
		}
	}
	else {
		if (label == nullptr) std::fill(Xs[0], Xs[0] + layerSizes[0] * batchSize, 0.0f); // or random ? TODO
		else std::copy(label, label + layerSizes[0] * batchSize, Xs[0]);

		std::copy(datapoint, datapoint + layerSizes[L] * batchSize, Xs[L]);
	}



	for (int l = 1; l < L; l++)
	{
		float f = powf((float)layerSizes[l], -.5f);
		for (int j = 0; j < batchSize * layerSizes[l]; j++) Xs[l][j] = NORMAL_01 * f;
		//std::fill(Xs[l], Xs[l] + layerSizes[l] * batchSize, 0.0f);

	}


#ifdef ADAM
	for (int l = 1; l < L+1; l++)
	{
		int s = layerSizes[l - 1] * layerSizes[l];
		std::fill(mW[l], mW[l] + s, 0.0f);
		std::fill(vW[l], vW[l] + s, 0.0f);
		std::fill(mB[l-1], mB[l-1] + layerSizes[l-1], 0.0f);
		std::fill(vB[l-1], vB[l-1] + layerSizes[l-1], 0.0f);
	}
#endif
}

void PCNN::computeEpsilons(bool training)
{
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
					mu += thetas[i+1][matID] * buffer1[k];
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

float PCNN::computeEnergy(bool training)
{
	float E = 0.0f;

	computeEpsilons(training);

	// epsilon 0 is relevant iff the network is: 
	// 
	//  - in reversed order
	// 
	// (X)OR
	// 
	// - in normal order, and we are in a supervised
	// phase, i.e. x0 is fixed to the target label.
	int first_l =  reversedOrder || 
				   (!reversedOrder && training)
		? 0 : 1;

	// epsilon L exists only in reversed order, while training.
	int last_l = (reversedOrder && training) ? L+1 : L;

	for (int l = first_l; l < last_l; l++) {
		for (int dp = 0; dp < batchSize; dp++) {
			for (int i = 0; i < layerSizes[l]; i++) {
				E += powf(epsilons[l][dp * layerSizes[l] + i], 2.0f);
			}
		}
	}

	return E/(float)batchSize;
}



void PCNN::learn(float tlr, float regularization, float beta1, float beta2) {
	computeEpsilons(true); // theta updates only occur when there is some sort of ground truth available.

	float normalization = 1.0f / (float)batchSize; // not already multiplied by tlr to avoid numerical instabilities (small values)


#ifdef ADAM
	float b1f = 1.0f / (1.0f - beta1);
	float b2f = 1.0f / (1.0f - beta2);
	constexpr float ADAM_epsilon = .0000001f;
#endif


	for (int l = 1; l < L+1; l++)
	{

#ifdef BIAS
		std::fill(biasesAccumulators[l - 1], biasesAccumulators[l - 1] + layerSizes[l - 1], 0.0f);
#endif
		std::fill(thetasAccumulators[l], thetasAccumulators[l] + layerSizes[l - 1] * layerSizes[l], 0.0f);

		for (int dp = 0; dp < batchSize; dp++)
		{
			for (int k = 0; k < layerSizes[l]; k++) {
				buffer1[k] = tanhf(Xs[l][dp * layerSizes[l] + k]);
			}

			for (int j = 0; j < layerSizes[l - 1]; j++) {
				for (int k = 0; k < layerSizes[l]; k++) {
					thetasAccumulators[l][j * layerSizes[l] + k] +=
						buffer1[k] * epsilons[l - 1][dp * layerSizes[l - 1] + j];
				}
#ifdef BIAS
				biasesAccumulators[l - 1][j] += epsilons[l - 1][dp * layerSizes[l - 1] + j];
#endif
			}
		}

		for (int j = 0; j < layerSizes[l - 1]; j++) {
			for (int k = 0; k < layerSizes[l]; k++) {
				thetasAccumulators[l][j * layerSizes[l] + k] *= normalization;
#ifdef ADAM
				mW[l][j * layerSizes[l] + k] = beta1 * mW[l][j * layerSizes[l] + k] +
					(1.0f - beta1) * thetasAccumulators[l][j * layerSizes[l] + k];
				float mhat = mW[l][j * layerSizes[l] + k] * b1f;

				vW[l][j * layerSizes[l] + k] = beta2 * vW[l][j * layerSizes[l] + k] +
					(1.0f - beta2) * powf(thetasAccumulators[l][j * layerSizes[l] + k], 2.0f);
				float vhat = vW[l][j * layerSizes[l] + k] * b2f;
				
				thetas[l][j * layerSizes[l] + k] = thetas[l][j * layerSizes[l] + k] * regularization +
					tlr * mhat * powf(vhat + ADAM_epsilon, -.5f);
#else
				thetas[l][j * layerSizes[l] + k] = thetas[l][j * layerSizes[l] + k] * regularization +
					(thetasAccumulators[l][j * layerSizes[l] + k] * tlr);
#endif
				
			}
#ifdef BIAS
#ifdef ADAM
			mB[l - 1][j] = beta1 * mB[l - 1][j] +
				(1.0f - beta1) * biasesAccumulators[l - 1][j];
			float mhat = mB[l - 1][j] * b1f;

			vB[l - 1][j] = beta2 * vB[l - 1][j] +
				(1.0f - beta2) * powf(biasesAccumulators[l - 1][j], 2.0f);
			float vhat = vB[l - 1][j] * b2f;

			biases[l - 1][j] = biases[l - 1][j] * regularization +
				tlr * mhat * powf(vhat + ADAM_epsilon, -.5f);
#else
			biases[l - 1][j] = biases[l - 1][j] * regularization +
				(biasesAccumulators[l - 1][j] * tlr) * normalization;
#endif
#endif
		}
	}
}



void PCNN::iPC_Interleaved_Forward_DataInXL(float tlr, float xlr, float regularization, float beta1, float beta2) 
{

	float normalization = 1.0f / (float)batchSize;

	for (int l = L - 1; l > 0; l--)
	{
#ifdef BIAS
		std::fill(biasesAccumulators[l], biasesAccumulators[l] + layerSizes[l], 0.0f);
#endif
		std::fill(thetasAccumulators[l + 1], thetasAccumulators[l + 1] + layerSizes[l] * layerSizes[l + 1], 0.0f);

		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset, e_offset, matID;


			// epsilon l
			x_offset = dp * layerSizes[l + 1];
			e_offset = dp * layerSizes[l];
			for (int j = 0; j < layerSizes[l + 1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l + 1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - mu;
			}


			// epsilon l - 1. could be precomputed for all layers out of the loop.
			x_offset = dp * layerSizes[l];
			e_offset = dp * layerSizes[l - 1];
			for (int j = 0; j < layerSizes[l]; j++) {
				buffer1[j] = tanhf(Xs[l][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l - 1]; i++) {
#ifdef BIAS
				float mu = biases[l - 1][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l]; j++) {
					mu += thetas[l][matID] * buffer1[j];
					matID++;
				}
				epsilons[l - 1][e_offset + i] = Xs[l - 1][e_offset + i] - mu;
			}


			// compute and add deltaX l
			for (int j = 0; j < layerSizes[l]; j++) {
				float s = 0.0f;
				for (int k = 0; k < layerSizes[l - 1]; k++) {
					s += thetas[l][j + k * layerSizes[l]] * epsilons[l - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][x_offset + j]), 2.0f);
				Xs[l][x_offset + j] += xlr * (dfx * s - epsilons[l][x_offset + j]);
			}

			// recompute epsilon l
			x_offset = dp * layerSizes[l + 1];
			e_offset = dp * layerSizes[l];
			for (int j = 0; j < layerSizes[l + 1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l + 1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - mu;
			}

			for (int j = 0; j < layerSizes[l]; j++) {
				for (int k = 0; k < layerSizes[l + 1]; k++) {
					thetasAccumulators[l + 1][j * layerSizes[l + 1] + k] +=
						buffer1[k] * epsilons[l][dp * layerSizes[l] + j];
				}
#ifdef BIAS
				biasesAccumulators[l][j] += epsilons[l][dp * layerSizes[l] + j];
#endif
			}
		}

		// apply delta theta l+1:
		for (int j = 0; j < layerSizes[l - 1]; j++) {
			for (int k = 0; k < layerSizes[l]; k++) {
				thetasAccumulators[l][j * layerSizes[l] + k] *= normalization;
				thetas[l][j * layerSizes[l] + k] = thetas[l][j * layerSizes[l] + k] * regularization +
					(thetasAccumulators[l][j * layerSizes[l] + k] * tlr);

			}
#ifdef BIAS
			biases[l - 1][j] = biases[l - 1][j] * regularization +
				(biasesAccumulators[l - 1][j] * tlr) * normalization;
#endif
		}
	}
}

void PCNN::infer_Simultaneous_DataInXL(float xlr, bool training)
{
	computeEpsilons(training);

	// update xl for l in [1, L-1]
	for (int i = 1; i < L; i++)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{

			int e_offset = dp * layerSizes[i-1];
			int x_offset = dp * layerSizes[i];

			for (int j = 0; j < layerSizes[i]; j++) {
				float s = 0.0f;
				for (int k = 0; k < layerSizes[i-1]; k++) {
					s += thetas[i][j + k * layerSizes[i]] * epsilons[i - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[i][x_offset + j]), 2.0f);
				Xs[i][x_offset + j] += xlr * (dfx * s - epsilons[i][x_offset + j]);
			}
		}
	}

	if (!training)  // x0 is updated if not fixed at the label.
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset = layerSizes[0] * dp;
			for (int i = 0; i < layerSizes[0]; i++) {
				Xs[0][x_offset + i] -= xlr * epsilons[0][x_offset + i];
			}
		}
	}
}

void PCNN::infer_Forward_DataInXL(float xlr, bool training)
{

	for (int l = L-1; l > 0; l--)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset, e_offset, matID;


			// epsilon l
			x_offset = dp * layerSizes[l+1];
			e_offset = dp * layerSizes[l];
			for (int j = 0; j < layerSizes[l+1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l+1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - mu;
			}
			

			// epsilon l - 1. could be precomputed for all layers out of the loop.
			x_offset = dp * layerSizes[l];
			e_offset = dp * layerSizes[l-1];
			for (int j = 0; j < layerSizes[l]; j++) {
				buffer1[j] = tanhf(Xs[l][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l-1]; i++) {
#ifdef BIAS
				float mu = biases[l-1][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l]; j++) {
					mu += thetas[l][matID] * buffer1[j];
					matID++;
				}
				epsilons[l-1][e_offset + i] = Xs[l-1][e_offset + i] - mu;
			}


			// compute and add deltaX l
			for (int j = 0; j < layerSizes[l]; j++) {
				float s = 0.0f;
				for (int k = 0; k < layerSizes[l - 1]; k++) {
					s += thetas[l][j + k * layerSizes[l]] * epsilons[l - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][x_offset + j]), 2.0f);
				Xs[l][x_offset + j] += xlr * (dfx * s - epsilons[l][x_offset + j]);
			}
		}
	}

	if (!training)  // x0 is updated if not fixed at the label.
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			//epsilon 0
			int x_offset = dp * layerSizes[1];
			int e_offset = dp * layerSizes[0];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][x_offset + j]); // f(x1)
			}
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}
				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - mu;
			}

			// x0 += dx0
			x_offset = layerSizes[0] * dp;
			for (int i = 0; i < layerSizes[0]; i++) {
				Xs[0][x_offset + i] -= xlr * epsilons[0][x_offset + i];
			}
		}
	}
}

void PCNN::infer_Backward_DataInXL(float xlr, bool training) 
{
	// x0 is updated if not fixed at the label.
	if (!training)  
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			// prepare data for epsilon 0
			int x_offset = dp * layerSizes[1];
			int e_offset = dp * layerSizes[0];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][x_offset + j]); // f(x1)
			}
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}
				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - mu;

				// x0 += dx0
				Xs[0][e_offset + i] -= xlr * epsilons[0][e_offset + i];
				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - mu;
			}
		}
	}
	// otherwise at least compute epsilon 0 to prepare x1's update.
	else 
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			// prepare data for epsilon 0
			int x_offset = dp * layerSizes[1];
			int e_offset = dp * layerSizes[0];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][x_offset + j]); // f(x1)
			}
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}

				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - mu;
			}
		}
	}

	for (int l = 1; l < L; l++)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset, e_offset, matID;


			// prepare data for epsilon l
			x_offset = dp * layerSizes[l + 1];
			e_offset = dp * layerSizes[l];
			for (int j = 0; j < layerSizes[l + 1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {

#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l + 1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - mu;
		

				// compute and add deltaX l
			
				float s = 0.0f;
				for (int j = 0; j < layerSizes[l - 1]; j++) {
					s += thetas[l][i + j * layerSizes[l]] * epsilons[l - 1][e_offset + j];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][x_offset + i]), 2.0f);
				Xs[l][x_offset + i] += xlr * (dfx * s - epsilons[l][x_offset + i]);

				// precompute epsilon l for the next loop iteration.
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - mu;
			}
		}
	}
}


void PCNN::iPC_Interleaved_Forward_DataInX0(float tlr, float xlr, float regularization, float beta1, float beta2) 
{
	for (int l = 0; l < L; l++) 
	{
		for (int dp = 0; dp < batchSize; dp++)
		{

			
			int offset = dp * layerSizes[l + 1];
			for (int j = 0; j < layerSizes[l + 1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][offset + j]); // f(xl+1)
			}


			offset = dp * layerSizes[l];
			int p_offset = dp * layerSizes[l - 1];
			int matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l + 1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][offset + i] = Xs[l][offset + i] - mu;

				Xs[l][offset + i] -= xlr * epsilons[l][offset + i];

				epsilons[l][offset + i] = Xs[l][offset + i] - mu;
			}
		}
	}
}

void PCNN::infer_Simultaneous_DataInX0(float xlr, bool training, int* corruptedIndices)
{
	computeEpsilons(training);


	//float* alpha = new float[(L + 1) * batchSize];
	//float* beta = new float[(L + 1) * batchSize];

	//for (int l = 0; l < L+1; l++) {
	//	float f = 1.0f / layerSizes[l];
	//	for (int dp = 0; dp < batchSize; dp++)
	//	{
	//		int offset = dp * layerSizes[l];
	//		float s = 0.0f;
	//		for (int i = 0; i < layerSizes[l]; i++) {
	//			s += powf(epsilons[l][offset+i], 2.0f);
	//		}
	//		s *= f;

	//		// temporarily stores error norms
	//		alpha[l * batchSize + dp] = s;
	//	}
	//}

	//const float f = 1.5f;
	//const float inv_f = 1.0f / f;
	//for (int l = 0; l < L; l++) {
	//	for (int dp = 0; dp < batchSize; dp++)
	//	{
	//		// for clarity
	//		float epsilonl = alpha[l * batchSize + dp];
	//		float epsilonlplus1 = alpha[(l + 1) * batchSize + dp] + .000001f;

	//		alpha[l * batchSize + dp] = xlr / std::clamp(epsilonl / epsilonlplus1, inv_f, f);
	//		beta[(l+1) * batchSize + dp] = alpha[l * batchSize + dp];
	//	}
	//}

	//std::copy(&beta[L * batchSize], &beta[L * batchSize] + batchSize, &alpha[L * batchSize]);


	// delta xl for l in [1,L]. epsilons being precomputed, 
	// order does not matter. All could happen in parallel.
	for (int l = 1; l < L+1; l++) // L+(training?0:1) fixes xL at the label (if correctly initialized) during training.
	{
		for (int dp = 0; dp < batchSize; dp++)
		{

			int e_offset = dp * layerSizes[l - 1];
			int x_offset = dp * layerSizes[l];

			for (int j = 0; j < layerSizes[l]; j++) {
				float s = 0.0f;
				for (int k = 0; k < layerSizes[l - 1]; k++) {
					s += thetas[l][j + k * layerSizes[l]] * epsilons[l - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][x_offset + j]), 2.0f);
				Xs[l][x_offset + j] += xlr * (dfx * s - epsilons[l][x_offset + j]);
				//Xs[l][x_offset + j] += beta[l * batchSize + dp] * dfx * s - alpha[l * batchSize + dp] * epsilons[l][x_offset + j];
			}
		}
	}

	// delta x0 is to be considered iff x0 is fixed 
	// (at least partially) to a corrupted / partial input.
	if (corruptedInput && !training) {
		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset = dp * layerSizes[0];

			for (int j = 0; j < layerSizes[0]; j++) {
				if (corruptedIndices[j] == 1) [[unlikely]] continue;
				float dfx = 1.0f - powf(tanhf(Xs[0][x_offset + j]), 2.0f);
				Xs[0][x_offset + j] -= xlr * epsilons[0][x_offset + j];
				//Xs[0][x_offset + j] -= alpha[dp] * epsilons[0][x_offset + j];
			}
		}
	}
}

void PCNN::infer_Forward_DataInX0(float xlr, bool training, int* corruptedIndices)
{

	// apply delta x0 if incomplete / corrupted input, ( so not during the learning phase)
	if (corruptedInput && !training) {
		for (int dp = 0; dp < batchSize; dp++)
		{

			int offset = dp * layerSizes[1];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][offset + j]); // f(x1)
			}

			offset = dp * layerSizes[0];
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
				if (corruptedIndices[i] == 1) [[unlikely]] continue;

#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}
				epsilons[0][offset + i] = Xs[0][offset + i] - mu;

				// x0 += dx0
				Xs[0][offset + i] -= xlr * epsilons[0][offset + i];

				epsilons[0][offset + i] = Xs[0][offset + i] - mu;
			}
		}
	}
	// otherwise at least compute epsilon 0 to prepare x1's update
	else 
	{ 
		for (int dp = 0; dp < batchSize; dp++)
		{

			int offset = dp * layerSizes[1];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][offset + j]); // f(x1)
			}

			offset = dp * layerSizes[0];
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}

				epsilons[0][offset + i] = Xs[0][offset + i] - mu;
			}
		}
	}

	// delta xl for l in [1, L-1], in ascending order.
	// epsilon l is recomputed at the end to be usable by delta l+1.
	for (int l = 1; l < L; l++)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int offset = dp * layerSizes[l+1];
			for (int j = 0; j < layerSizes[l+1]; j++) {
				buffer1[j] = tanhf(Xs[l+1][offset + j]); // f(xl+1)
			}


			offset = dp * layerSizes[l];
			int p_offset = dp * layerSizes[l-1];
			int matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l+1]; j++) {
					mu += thetas[l+1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][offset + i] = Xs[l][offset + i] - mu;


				float s = 0.0f;
				for (int k = 0; k < layerSizes[l - 1]; k++) {
					s += thetas[l][i + k * layerSizes[l]] * epsilons[l - 1][p_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][offset + i]), 2.0f);
				Xs[l][offset + i] += xlr * (dfx * s - epsilons[l][offset + i]);

				epsilons[l][offset + i] = Xs[l][offset + i] - mu;
			}
		}
	}

	// epsilon L
	if (!training) {
		std::fill(epsilons[L], epsilons[L] + batchSize * layerSizes[L], 0.0f);
	}
	else {
		for (int idp = 0; idp < batchSize * layerSizes[L]; idp++)
		{
			epsilons[L][idp] = Xs[L][idp] - muL[idp];
		}
	}
	// delta xL
	for (int dp = 0; dp < batchSize; dp++)
	{

		int offset = dp * layerSizes[L];
		int p_offset = dp * layerSizes[L - 1];
		int matID = 0;
		for (int i = 0; i < layerSizes[L]; i++) {
			
			float s = 0.0f;
			for (int k = 0; k < layerSizes[L - 1]; k++) {
				s += thetas[L][i + k * layerSizes[L]] * epsilons[L - 1][p_offset + k];
			}

			float dfx = 1.0f - powf(tanhf(Xs[L][offset + i]), 2.0f);
			Xs[L][offset + i] += xlr * (dfx * s - epsilons[L][offset + i]);
		}
	}
}

void PCNN::infer_Backward_DataInX0(float xlr, bool training, int* corruptedIndices)
{
	// so that xl for l in [1, L] can use epsilon l-1, and xL can use epsilonL
	computeEpsilons(training);

	// delta xL
	for (int dp = 0; dp < batchSize; dp++)
	{

		int offset = dp * layerSizes[L];
		int p_offset = dp * layerSizes[L - 1];
		int matID = 0;
		for (int i = 0; i < layerSizes[L]; i++) {

			float s = 0.0f;
			for (int k = 0; k < layerSizes[L - 1]; k++) {
				s += thetas[L][i + k * layerSizes[L]] * epsilons[L - 1][p_offset + k];
			}

			float dfx = 1.0f - powf(tanhf(Xs[L][offset + i]), 2.0f);
			Xs[L][offset + i] += xlr * (dfx * s - epsilons[L][offset + i]);
		}
	}

	// delta xl for l in [1, L-1], decreasing order.
	for (int l = L-1; l > 0; l--)
	{
		for (int dp = 0; dp < batchSize; dp++)
		{
			int offset = dp * layerSizes[l + 1];
			for (int j = 0; j < layerSizes[l + 1]; j++) {
				buffer1[j] = tanhf(Xs[l + 1][offset + j]); // f(xl+1)
			}


			offset = dp * layerSizes[l];
			int p_offset = dp * layerSizes[l - 1];
			int matID = 0;
			for (int i = 0; i < layerSizes[l]; i++) {
#ifdef BIAS
				float mu = biases[l][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					mu += thetas[l + 1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][offset + i] = Xs[l][offset + i] - mu;


				float s = 0.0f;
				for (int k = 0; k < layerSizes[l - 1]; k++) {
					s += thetas[l][i + k * layerSizes[l]] * epsilons[l - 1][p_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][offset + i]), 2.0f);
				Xs[l][offset + i] += xlr * (dfx * s - epsilons[l][offset + i]);

				epsilons[l][offset + i] = Xs[l][offset + i] - mu;
			}
		}
	}

	// apply delta x0 if incomplete / corrupted input, at runtime only
	if (corruptedInput && !training) {
		for (int dp = 0; dp < batchSize; dp++)
		{

			int offset = dp * layerSizes[1];
			for (int j = 0; j < layerSizes[1]; j++) {
				buffer1[j] = tanhf(Xs[1][offset + j]); // f(x1)
			}

			offset = dp * layerSizes[0];
			int matID = 0;
			for (int i = 0; i < layerSizes[0]; i++) {
				if (corruptedIndices[i] == 1) [[unlikely]] continue;
#ifdef BIAS
				float mu = biases[0][i];
#else
				float mu = 0.0f;
#endif
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}
				epsilons[0][offset + i] = Xs[0][offset + i] - mu;

				// x0 += dx0
				Xs[0][offset + i] -= xlr * epsilons[0][offset + i];

				//epsilons[0][offset + i] = Xs[0][offset + i] - mu; unnecessary
			}
		}
	}
}
