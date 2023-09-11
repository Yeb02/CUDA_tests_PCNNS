#include "PCNN.h"


PCNN::PCNN(int _L, int* _lS, int _dS, bool _reversedOrder, bool _corruptedInput) :
	L(_L), layerSizes(_lS), batchSize(_dS), reversedOrder(_reversedOrder), corruptedInput(_corruptedInput)
{
	Xs.resize(L);
	epsilons.resize(L);
	thetas.resize(L);

	if (reversedOrder) muL = new float[batchSize * layerSizes[L]];
	else muL = nullptr;

	Xs[0] = new float[batchSize * layerSizes[0]];
	epsilons[0] = new float[batchSize * layerSizes[0]];
	thetas[0] = nullptr;

	int lSmax = layerSizes[0];
	for (int i = 1; i < L; i++) 
	{
		Xs[i] = new float[batchSize * layerSizes[i]];
		epsilons[i] = new float[batchSize * layerSizes[i]];

		int tS = layerSizes[i - 1] * layerSizes[i];
		thetas[i] = new float[tS];
		float f = powf((float)tS, -.5f); // HE init
		for (int j = 0; j < tS; j++) thetas[i][j] = NORMAL_01 * f;

		if (layerSizes[i] > lSmax) lSmax = layerSizes[i];
	}

	buffer1 = new float[lSmax];
	buffer2 = new float[lSmax];

	if (reversedOrder) output = Xs[L];
	else output = Xs[0];

}

void PCNN::initXs(float* datapoint, float* label)
{
	

	if (reversedOrder) {
		std::copy(datapoint, datapoint + layerSizes[0] * batchSize, Xs[0]);
		
		if (label == nullptr) std::fill(muL, muL + layerSizes[L] * batchSize, 0.0f);
		else std::copy(label, label + layerSizes[L] * batchSize, muL);

		std::fill(Xs[L], Xs[L] + batchSize * layerSizes[L], 0.0f);
	}
	else {
		if (label == nullptr) std::fill(Xs[0], Xs[0] + layerSizes[0] * batchSize, 0.0f); // or random ? TODO
		else std::copy(label, label + layerSizes[0] * batchSize, Xs[0]);

		std::copy(datapoint, datapoint + layerSizes[L] * batchSize, Xs[L]);
	}

	for (int i = 1; i < L; i++)
	{
		float f = powf((float)layerSizes[i], -.5f);
		for (int j = 0; j < batchSize * layerSizes[i]; j++) Xs[i][j] = NORMAL_01 * f;
	}
}

void PCNN::computeEpsilons()
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
				float s = 0.0f;
				for (int k = 0; k < layerSizes[i + 1]; k++) {
					s += thetas[i+1][matID] * buffer1[k];
					matID++;
				}
				epsilons[i][offset + j] = Xs[i][offset + j] - s;
			}
		}

	}

	//if reversedOrder epsilon L must be computed
	if (reversedOrder) {
		for (int dp = 0; dp < batchSize; dp++)
		{
			int offset = dp * layerSizes[L];
			for (int j = 0; j < layerSizes[L]; j++) {
				epsilons[L][offset + j] = Xs[L][offset + j] - muL[offset + j];
			}
		}
	}
}

float PCNN::computeEnergy()
{
	float E = 0.0f;

	computeEpsilons();

	// epsilon 0 is relevant only in reversed order,
	// and iff we are feeding corrupted / partial inputs 
	// to x0. When it is the case the "inference" functions
	// must implement the delta x0 change at runtime.
	int first_l = reversedOrder && corruptedInput ? 0 : 1;

	// epsilon L exists only in reversed order.
	int last_l = reversedOrder ? L : L-1;

	for (int l = first_l; l < last_l; l++) {
		for (int dp = 0; dp < batchSize; dp++) {
			for (int i = 0; i < layerSizes[l]; i++) {
				E += powf(epsilons[l][dp * layerSizes[l] + i], 2.0f);
			}
		}
	}

	return E;
}


void PCNN::infer_Simultaneous_DataInXL(float xlr, bool training)
{
	computeEpsilons();

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
	for (int l = L-2; l > 0; l--)
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
				float s = 0.0f;
				for (int j = 0; j < layerSizes[l + 1]; j++) {
					s += thetas[l+1][matID] * buffer1[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - s;
			}
			

			// epsilon l - 1
			x_offset = dp * layerSizes[l];
			e_offset = dp * layerSizes[l-1];
			for (int j = 0; j < layerSizes[l]; j++) {
				buffer1[j] = tanhf(Xs[l][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < layerSizes[l-1]; i++) {
				float s = 0.0f;
				for (int j = 0; j < layerSizes[l]; j++) {
					s += thetas[l][matID] * buffer1[j];
					matID++;
				}
				epsilons[l-1][e_offset + i] = Xs[l-1][e_offset + i] - s;
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
				float s = 0.0f;
				for (int j = 0; j < layerSizes[1]; j++) {
					s += thetas[1][matID] * buffer1[j];
					matID++;
				}
				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - s;
			}

			// x0 += dx0
			x_offset = layerSizes[0] * dp;
			for (int i = 0; i < layerSizes[0]; i++) {
				Xs[0][x_offset + i] -= xlr * epsilons[0][x_offset + i];
			}
		}
	}
}

void PCNN::learn_DataInXL(float tlr, float regularization)
{
	computeEpsilons();

	for (int l = 1; l < L; l++)
	{
		for (int j = 0; j < layerSizes[l - 1]; j++) {
			for (int k = 0; k < layerSizes[l]; k++) {
				float s = 0.0f;
				for (int dp = 0; dp < batchSize; dp++)
				{
					s += Xs[l][dp * layerSizes[l] + k] * epsilons[l - 1][dp * layerSizes[l - 1] + j];
				}
				thetas[l][j * layerSizes[l] + k] = tlr * s + regularization * thetas[l][j * layerSizes[l] + k];
			}
		}
	}
}


void PCNN::infer_Simultaneous_DataInX0(float xlr, bool training)
{
	computeEpsilons();

	if (!training) {
		std::fill(epsilons[L], epsilons[L] + batchSize * layerSizes[L], 0.0f);
	}

	// delta xl for l in [1,L]. delta x0 is to be considered iff x0 is fixed 
	// (at least partially) to a corrupted / partial input.
	for (int l = 1; l < L; l++)
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
			}
		}
	}

	if (corruptedInput && !training) {
		for (int dp = 0; dp < batchSize; dp++)
		{
			int x_offset = dp * layerSizes[0];

			for (int j = 0; j < layerSizes[0]; j++) {

				float dfx = 1.0f - powf(tanhf(Xs[0][x_offset + j]), 2.0f);
				Xs[0][x_offset + j] -= xlr * epsilons[0][x_offset + j];
			}
		}
	}
}

void PCNN::infer_Forward_DataInX0(float xlr, bool training) 
{

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
				float mu = 0.0f;
				for (int j = 0; j < layerSizes[1]; j++) {
					mu += thetas[1][matID] * buffer1[j];
					matID++;
				}
				buffer2[i] = mu;
				epsilons[0][offset + i] = Xs[0][offset + i] - mu;

				// x0 += dx0
				Xs[0][offset + i] -= xlr * epsilons[0][offset + i];

				epsilons[0][offset + i] = Xs[0][offset + i] - mu;
			}
		}
	}

	// delta xl for l in [1, L-1]
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
				float mu = 0.0f;
				for (int j = 0; j < layerSizes[l+1]; j++) {
					mu += thetas[l+1][matID] * buffer1[j];
					matID++;
				}
				buffer2[i] = mu;
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

	// delta xL
	if (!training) {
		std::fill(epsilons[L], epsilons[L] + batchSize * layerSizes[L], 0.0f);
	}
	else {
		for (int idp = 0; idp < batchSize * layerSizes[L]; idp++)
		{
			epsilons[L][idp] = Xs[L][idp] - muL[idp];
		}
	}
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

// same as learn_DataInXL
void PCNN::learn_DataInX0(float tlr, float regularization) {
	computeEpsilons();

	for (int l = 1; l < L; l++)
	{
		for (int j = 0; j < layerSizes[l - 1]; j++) {
			for (int k = 0; k < layerSizes[l]; k++) {
				float s = 0.0f;
				for (int dp = 0; dp < batchSize; dp++)
				{
					s += Xs[l][dp * layerSizes[l] + k] * epsilons[l - 1][dp * layerSizes[l - 1] + j];
				}
				thetas[l][j * layerSizes[l] + k] = tlr * s + regularization * thetas[l][j * layerSizes[l] + k];
			}
		}
	}
}
