#include "PCNN.h"


PCNN::PCNN(int _nL, int* _lS, int _dS, bool _reversedOrder) : 
	nL(_nL), lS(_lS), dS(_dS), reversedOrder(_reversedOrder)
{
	Xs.resize(nL);
	epsilons.resize(nL);
	thetas.resize(nL);

	if (reversedOrder) muL = new float[dS * lS[nL - 1]];
	else muL = nullptr;

	Xs[0] = new float[dS * lS[0]];
	epsilons[0] = new float[dS * lS[0]];
	thetas[0] = nullptr;

	int lSmax = lS[0];
	for (int i = 1; i < nL; i++) 
	{
		Xs[i] = new float[dS * lS[i]];
		epsilons[i] = new float[dS * lS[i]];

		int tS = lS[i - 1] * lS[i];
		thetas[i] = new float[tS];
		float f = powf((float)tS, -.5f); // HE init
		for (int j = 0; j < tS; j++) thetas[i][j] = NORMAL_01 * f;

		if (lS[i] > lSmax) lSmax = lS[i];
	}

	if (reversedOrder) output = Xs[nL-1];
	else output = Xs[0];

	buffer = new float[lSmax];
}

void PCNN::initXs(float* datapoint, float* label)
{
	

	if (reversedOrder) {
		std::copy(datapoint, datapoint + lS[0] * dS, Xs[0]);
		
		if (label == nullptr) std::fill(muL, muL + lS[0] * dS, 0.0f);
		else std::copy(label, label + lS[nL - 1] * dS, muL);

		std::fill(Xs[nL - 1], Xs[nL - 1] + dS * lS[nL - 1], 0.0f);
	}
	else {
		if (label == nullptr) std::fill(Xs[0], Xs[0] + lS[0] * dS, 0.0f); // or random ? TODO
		else std::copy(label, label + lS[0] * dS, Xs[0]);

		std::copy(datapoint, datapoint + lS[nL - 1] * dS, Xs[nL - 1]);
	}

	for (int i = 1; i < nL-1; i++)
	{
		float f = powf((float)lS[i], -.5f);
		for (int j = 0; j < dS * lS[i]; j++) Xs[i][j] = NORMAL_01 * f;
	}
}

void PCNN::computeEpsilons()
{
	for (int i = 0; i < nL - 1; i++)
	{
		for (int dp = 0; dp < dS; dp++)
		{
			int offset = dp * lS[i + 1];
			for (int k = 0; k < lS[i + 1]; k++) {
				buffer[k] = tanhf(Xs[i + 1][offset + k]);
			}

			int matID = 0;
			offset = dp * lS[i];
			for (int j = 0; j < lS[i]; j++) {
				float s = 0.0f;
				for (int k = 0; k < lS[i + 1]; k++) {
					s += thetas[i+1][matID] * buffer[k];
					matID++;
				}
				epsilons[i][offset + j] = Xs[i][offset + j] - s;
			}
		}

	}

	//epsilon L must be computed
	if (reversedOrder) {
		for (int dp = 0; dp < dS; dp++)
		{
			int offset = dp * lS[nL - 1];
			for (int j = 0; j < lS[nL - 1]; j++) {
				epsilons[nL - 1][offset + j] = Xs[nL - 1][offset + j] - muL[offset + j];
			}
		}
	}
}



void PCNN::infer_Simultaneous_DataInXL(float xlr, bool training)
{
	computeEpsilons();

	for (int i = 1; i < nL - 1; i++)
	{
		for (int dp = 0; dp < dS; dp++)
		{

			int e_offset = dp * lS[i-1];
			int x_offset = dp * lS[i];

			for (int j = 0; j < lS[i]; j++) {
				float s = 0.0f;
				for (int k = 0; k < lS[i-1]; k++) {
					s += thetas[i][j + k * lS[i]] * epsilons[i - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[i][x_offset + j]), 2.0f);
				Xs[i][x_offset + j] += xlr * (dfx * s - epsilons[i][x_offset + j]);
			}
		}
	}

	if (!training)  // x0 is updated if not fixed at the label.
	{
		for (int dp = 0; dp < dS; dp++)
		{
			int x_offset = lS[0] * dp;
			for (int i = 0; i < lS[0]; i++) {
				Xs[0][x_offset + i] -= xlr * epsilons[0][x_offset + i];
			}
		}
	}
}

void PCNN::learn_Simultaneous_DataInXL(float tlr)
{
	computeEpsilons();

	for (int i = 1; i < nL; i++)
	{
		for (int j = 0; j < lS[i-1]; j++) {
			for (int k = 0; k < lS[i]; k++) {
				float s = 0.0f;
				for (int dp = 0; dp < dS; dp++)
				{
					s += Xs[i][dp * lS[i] + k] * epsilons[i - 1][dp * lS[i - 1] + j];
				}
				thetas[i][j * lS[i] + k] += tlr * s;
			}
		}
	}
}



void PCNN::infer_Forward_DataInXL(float xlr, bool training)
{
	for (int l = nL-2; l > 0; l--)
	{
		for (int dp = 0; dp < dS; dp++)
		{
			int x_offset, e_offset, matID;


			// epsilon l
			x_offset = dp * lS[l+1];
			e_offset = dp * lS[l];
			for (int j = 0; j < lS[l+1]; j++) {
				buffer[j] = tanhf(Xs[l + 1][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < lS[l]; i++) {
				float s = 0.0f;
				for (int j = 0; j < lS[l + 1]; j++) {
					s += thetas[l+1][matID] * buffer[j];
					matID++;
				}
				epsilons[l][e_offset + i] = Xs[l][e_offset + i] - s;
			}
			

			// epsilon l - 1
			x_offset = dp * lS[l];
			e_offset = dp * lS[l-1];
			for (int j = 0; j < lS[l]; j++) {
				buffer[j] = tanhf(Xs[l][x_offset + j]); // f(x l+1)
			}
			matID = 0;
			for (int i = 0; i < lS[l-1]; i++) {
				float s = 0.0f;
				for (int j = 0; j < lS[l]; j++) {
					s += thetas[l][matID] * buffer[j];
					matID++;
				}
				epsilons[l-1][e_offset + i] = Xs[l-1][e_offset + i] - s;
			}


			// compute and add deltaX l
			for (int j = 0; j < lS[l]; j++) {
				float s = 0.0f;
				for (int k = 0; k < lS[l - 1]; k++) {
					s += thetas[l][j + k * lS[l]] * epsilons[l - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[l][x_offset + j]), 2.0f);
				Xs[l][x_offset + j] += xlr * (dfx * s - epsilons[l][x_offset + j]);
			}
		}
	}

	if (!training)  // x0 is updated if not fixed at the label.
	{
		for (int dp = 0; dp < dS; dp++)
		{
			//epsilon 0
			int x_offset = dp * lS[1];
			int e_offset = dp * lS[0];
			for (int j = 0; j < lS[1]; j++) {
				buffer[j] = tanhf(Xs[1][x_offset + j]); // f(x1)
			}
			int matID = 0;
			for (int i = 0; i < lS[0]; i++) {
				float s = 0.0f;
				for (int j = 0; j < lS[1]; j++) {
					s += thetas[1][matID] * buffer[j];
					matID++;
				}
				epsilons[0][e_offset + i] = Xs[0][e_offset + i] - s;
			}

			// x0 += dx0
			x_offset = lS[0] * dp;
			for (int i = 0; i < lS[0]; i++) {
				Xs[0][x_offset + i] -= xlr * epsilons[0][x_offset + i];
			}
		}
	}
}


void PCNN::infer_Simultaneous_DataInX0(float xlr, bool training)
{
	computeEpsilons();

	if (!training) {
		std::fill(epsilons[nL - 1], epsilons[nL - 1] + dS * lS[nL - 1], 0.0f);
	}

	// delta xl for l in [1,L]
	for (int i = 1; i < nL; i++)
	{
		for (int dp = 0; dp < dS; dp++)
		{

			int e_offset = dp * lS[i - 1];
			int x_offset = dp * lS[i];

			for (int j = 0; j < lS[i]; j++) {
				float s = 0.0f;
				for (int k = 0; k < lS[i - 1]; k++) {
					s += thetas[i][j + k * lS[i]] * epsilons[i - 1][e_offset + k];
				}

				float dfx = 1.0f - powf(tanhf(Xs[i][x_offset + j]), 2.0f);
				Xs[i][x_offset + j] += xlr * (dfx * s - epsilons[i][x_offset + j]);
			}
		}
	}
}


