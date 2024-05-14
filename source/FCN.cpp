#include "FCN.h"
#include <cmath> // copysign

#include <iomanip> // std::setprecision(5)


FCN::FCN(const int _nLayers, int* _sizes, int _datapointSize, float _weightRegularization, float _gradientStepSize) :
	nLayers(_nLayers), sizes(_sizes), gradientStepSize(_gradientStepSize), datapointSize(_datapointSize),
	weightRegularization(_weightRegularization)
{


	wx_variates = new float*[nLayers-1];
	wx_mean = new float*[nLayers-1];
	wx_precision = new float*[nLayers-1];
	
	bx_variates = new float*[nLayers];
	bx_mean = new float*[nLayers];
	bx_precision = new float*[nLayers];

	x = new float* [nLayers];
	fx = new float* [nLayers];
	epsilon = new float* [nLayers];
	tau = new float* [nLayers];

#ifdef DYNAMIC_PRECISIONS
	
#endif

	for (int i = 0; i < nLayers-1; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		float f = powf((float)sizes[i + 1], -.5f);

		wx_variates[i] = new float[s];
		wx_mean[i] = new float[s];
		wx_precision[i] = new float[s];
		
		
		std::fill(wx_mean[i], wx_mean[i] + s, .0f);
		std::fill(wx_precision[i], wx_precision[i] + s, 1.0f/f);

#ifdef DYNAMIC_PRECISIONS

#endif

		for (int j = 0; j < s; j++) {
			wx_variates[i][j] = NORMAL_01 * f * .3f;
		}
	}




	for (int i = 0; i < nLayers; i++)
	{

		bx_variates[i] = new float[sizes[i]];
		bx_mean[i] = new float[sizes[i]];
		bx_precision[i] = new float[sizes[i]];
		

		x[i] = new float[sizes[i]];
		fx[i] = new float[sizes[i]];
		epsilon[i] = new float[sizes[i]];
		tau[i] = new float[sizes[i]];

		std::fill(bx_mean[i], bx_mean[i] + sizes[i], .0f);
		std::fill(bx_precision[i], bx_precision[i] + sizes[i], 1.0f);

#ifdef DYNAMIC_PRECISIONS

#else
		std::fill(tau[i], tau[i] + sizes[i], 1.0f);
#endif

		for (int j = 0; j < sizes[i]; j++) {
			bx_variates[i][j] = NORMAL_01 * .3f;
		}

	}

	setXtoMAP(false);
}

FCN::~FCN() {

	for (int i = 0; i < nLayers; i++)
	{
		delete[] x[i];
		delete[] fx[i];
		delete[] epsilon[i];
		delete[] tau[i];

		delete[] bx_variates[i];
		delete[] bx_mean[i];
		delete[] bx_precision[i];

		if (i == nLayers - 1) break;

		delete[] wx_variates[i];
		delete[] wx_mean[i];
		delete[] wx_precision[i];
	}
}



#ifdef LABEL_IS_DATAPOINT
void FCN::sampleX(bool supervised)
{
	for (int i = nLayers - 1; i >= 1; i--)
	{

		for (int j = 0; j < sizes[i]; j++)
		{
			epsilon[i][j] = bx_variates[i][j];
		}
		if (i < nLayers - 1) {
			
			int id = 0;
			for (int j = 0; j < sizes[i]; j++)
			{
				for (int k = 0; k < sizes[i + 1]; k++)
				{
					epsilon[i][j] += wx_variates[i][id] * fx[i + 1][k];
					id++;
				}
			}
		}

		for (int j = 0; j < sizes[i]; j++)
		{
			x[i][j] = epsilon[i][j] + NORMAL_01 / tau[i][j];
			fx[i][j] = tanhf(x[i][j]);
		}
	}

	if (!supervised) {
		int offset_0 = sizes[0];
		int offset_1 = sizes[1];
		int id = 0;
		for (int j = datapointSize; j < sizes[0]; j++)
		{
			epsilon[0][j] = bx_variates[0][j];
			for (int k = 0; k < sizes[1]; k++)
			{
				epsilon[0][j] += wx_variates[0][id] * fx[1][k];
				id++;
			}
			x[0][j] = epsilon[0][j] + NORMAL_01 / tau[0][j];
		}
	}
}
#else
void FCN::sampleX(bool supervised)
{
	for (int i = nLayers - 1 - supervised; i >= 1; i--)
	{
		int offset_i = sizes[i] * threadID;

		for (int j = offset_i; j < sizes[i] + offset_i; j++)
		{
			epsilon[i][j] = b_variates[i][j];
		}
		if (i < nLayers - 1) {
			int offset_ip1 = sizes[i + 1] * threadID;
			int id = sizes[i] * sizes[i + 1] * threadID;
			for (int j = offset_i; j < sizes[i] + offset_i; j++)
			{
				for (int k = offset_ip1; k < sizes[i + 1] + offset_ip1; k++)
				{
					epsilon[i][j] += w_variates[i][id] * fx[i + 1][k];
					id++;
				}
			}
		}

		for (int j = offset_i; j < sizes[i] + offset_i; j++)
		{
			x[i][j] = epsilon[i][j] + NORMAL_01 / tau_variates[i][j];
			fx[i][j] = tanhf(x[i][j]);
		}
	}
}
#endif

#ifdef LABEL_IS_DATAPOINT
void FCN::setXtoMAP(bool supervised)
{
	for (int i = nLayers - 1; i >= 1; i--)
	{


		for (int j = 0; j < sizes[i]; j++)
		{
			epsilon[i][j] = bx_variates[i][j];
		}
		if (i < nLayers - 1) {
			int id = 0;
			for (int j = 0; j < sizes[i]; j++)
			{
				for (int k = 0; k < sizes[i + 1]; k++)
				{
					epsilon[i][j] += wx_variates[i][id] * fx[i + 1][k];
					id++;
				}
			}
		}

		for (int j = 0; j < sizes[i]; j++)
		{
			x[i][j] = epsilon[i][j];
			fx[i][j] = tanhf(x[i][j]);
		}
	}

	if (!supervised) {
		int id = 0;
		for (int j = datapointSize; j < sizes[0]; j++)
		{
			epsilon[0][j] = bx_variates[0][j];
			for (int k = 0; k < sizes[1]; k++)
			{
				epsilon[0][j] += wx_variates[0][id] * fx[1][k];
				id++;
			}
			x[0][j] = epsilon[0][j];
		}
	}
}
#else
void FCN::setXtoMAP(bool supervised)
{
	for (int i = nLayers - 1 - supervised; i >= 1; i--)
	{

		int offset_i = sizes[i] * threadID;

		for (int j = offset_i; j < offset_i + sizes[i]; j++)
		{
			epsilon[i][j] = b_variates[i][j];
		}
		if (i < nLayers - 1) {
			int offset_ip1 = sizes[i + 1] * threadID;
			int id = sizes[i + 1] * sizes[i] * threadID;
			for (int j = offset_i; j < offset_i + sizes[i]; j++)
			{
				for (int k = offset_ip1; k < offset_ip1 + sizes[i + 1]; k++)
				{
					epsilon[i][j] += w_variates[i][id] * fx[i + 1][k];
					id++;
				}
			}
		}

		for (int j = offset_i; j < offset_i + sizes[i]; j++)
		{
			x[i][j] = epsilon[i][j];
			fx[i][j] = tanhf(x[i][j]);
		}
	}
}
#endif



void FCN::computeEpsilons(int l)
{

	for (int j = 0; j < sizes[l]; j++)
	{
		epsilon[l][j] = bx_variates[l][j];
	}
	if (l < nLayers - 1) {
		int id = 0;
		for (int j = 0; j < sizes[l]; j++)
		{
			for (int k = 0; k < sizes[l + 1]; k++)
			{
				epsilon[l][j] += wx_variates[l][id] * fx[l + 1][k];
				id++;
			}
		}
	}
	for (int j = 0; j < sizes[l]; j++)
	{
		epsilon[l][j] = x[l][j] - epsilon[l][j];
	}
}


#ifdef LABEL_IS_DATAPOINT
void FCN::simultaneousAscentStep(bool supervised)
{
	for (int i = 0; i < nLayers; i++)
	{
		computeEpsilons(i);
	}


	// update variates. For simultaneity, if we want to avoid temporary variables, xl must be updated
	// before wl-1, but since wl-1's update depends on f(xl), we have to delay f(xl)'s update. bl's
	// update can happen anywhere.
	for (int i = 0; i < nLayers; i++)
	{

		// update xl but not f(xl), which is updated at the end of this loop's step
		if (i == 0 && !supervised) 
		{
			for (int j = datapointSize; j < sizes[i]; j++) {
				float gradx = - epsilon[i][j] * tau[i][j];
				x[i][j] += gradientStepSize * gradx;
			}
		}
		else if (i > 0) {

			for (int j = 0; j < sizes[i]; j++) {
				float gradx = - epsilon[i][j] * tau[i][j];
				float fprime = 1.0f - powf(fx[i][j], 2.f); // tanh prime
				float grad_acc = .0F;
				
				for (int k = 0; k < sizes[i-1]; k++) {
					grad_acc += epsilon[i - 1][k] * tau[i - 1][k] * wx_variates[i - 1][k*sizes[i] + j]; 
				}
				gradx += fprime * grad_acc;
				x[i][j] += gradientStepSize * gradx;
			}
		}
		


		// wl-1 then f(xl)
		if (i > 0)
		{
			int id = 0;
			for (int j = 0; j < sizes[i-1]; j++)
			{
				for (int k = 0; k < sizes[i]; k++)
				{
					float w = wx_variates[i-1][id];
					float gradw = - weightRegularization * w; 

					float ew = w - mu_w[i-1][id];
					gradw += -((.5f + alpha_w[i-1][id]) * ew) / ((1.f / nu_w[i-1][id] + 1.f) * beta_w[i-1][id] + .5f * ew * ew);
					gradw += 2.f * epsilon[i-1][j] * tau_variates[i-1][j] * fx[i][k];
					wx_variates[i-1][id + w_offset] += stepSize * gradw;
					id++;

				}
			}

			for (int j = 0; j < sizes[i]; j++) {
				fx[i][j] = tanhf(x[i][j]);
			}
		}

		// bl
		for (int j = 0; j < sizes[i]; j++)
		{
			float eb = b_variates[i][offj] - mu_b[i][j];
			float gradb = -((.5f + alpha_b[i][j]) * eb) / ((1.f / nu_b[i][j] + 1.f) * beta_b[i][j] + .5f * eb * eb);
			gradb += 2.f * epsilon[i][offj] * tau_variates[i][offj];
			b_variates[i][offj] += gradb * stepSize;
		}
	}
}
#else
TODO
#endif


void FCN::updateParameters() 
{
	float lr = 1.f / (float) N_THREADS; // Should be (on average) 1. 1/N_THREADS can be experimented with. TODO depends on each sample's energy
	for (int i = 0; i < nLayers; i++)
	{
		// w[i]
		if (i < nLayers - 1)
		{
			int id = 0;
			int offset = sizes[i] * sizes[i + 1];
			for (int j = 0; j < sizes[i]; j++)
			{
				for (int k = 0; k < sizes[i + 1]; k++)
				{
					float avg_variate = .0f;
					float totalWeight = .0f;
					for (int t = 0; t < N_THREADS; t++) {
						avg_variate += lr * w_variates[i][t * offset + id];
						totalWeight += lr;
					}
					avg_variate /= totalWeight;
					float emp_var = .0f;
					for (int t = 0; t < N_THREADS; t++) {
						emp_var += lr * powf(w_variates[i][t * offset + id] - avg_variate, 2.0f);
					}

					alpha_w[i][id] += .5f * totalWeight;
					beta_w[i][id] += .5f * (totalWeight * nu_w[i][id] / (nu_w[i][id] + totalWeight)) * powf(avg_variate - mu_w[i][id], 2.f) + .5f * emp_var;
					mu_w[i][id] = (nu_w[i][id] * mu_w[i][id] + totalWeight * avg_variate) / (nu_w[i][id] + totalWeight);
					nu_w[i][id] += totalWeight;

					id++;
				}
			}
		}

		// b[i]
		for (int j = 0; j < sizes[i]; j++)
		{
			float avg_variate = .0f;
			float totalWeight = .0f;
			for (int t = 0; t < N_THREADS; t++) {
				avg_variate += lr * b_variates[i][t * sizes[i] + j];
				totalWeight += lr;
			}
			avg_variate /= totalWeight;
			float emp_var = .0f;
			for (int t = 0; t < N_THREADS; t++) {
				emp_var += lr * powf(b_variates[i][t * sizes[i] + j] - avg_variate, 2.0f);
			}

			alpha_b[i][j] += .5f * totalWeight;
			beta_b[i][j] += .5f * (totalWeight * nu_b[i][j] / (nu_b[i][j] + totalWeight)) * powf(avg_variate - mu_b[i][j], 2.f) + .5f * emp_var;
			mu_b[i][j] = (nu_b[i][j] * mu_b[i][j] + totalWeight * avg_variate) / (nu_b[i][j] + totalWeight);
			nu_b[i][j] += totalWeight;
		}
	}
}


float FCN::computePerActivationEnergy()
{
	float E = 0.f;
	int n = 0;

	for (int i = 0; i < nLayers; i++)
	{
		n += sizes[i];
		computeEpsilons(i);
		for (int j = 0; j < sizes[i]; j++)
		{
			E += .5f * epsilon[i][j] * epsilon[i][j] * tau[i][j] - logf(tau[i][j]); // + constants
		}
	}

	return E/(float) n;
}

float FCN::computePerVariateEnergy()
{
	float E = 0.f;
	int n = 0;

	for (int i = 0; i < nLayers; i++)
	{
		n += sizes[i];
		computeEpsilons(i);
		for (int j = 0; j < sizes[i]; j++)
		{
			E += .5f * epsilon[i][j] * epsilon[i][j] * tau[i][j] - logf(tau[i][j]); // + constants
		}


		 // TODO
	}

	return E / (float)n;
}


void FCN::learn(float* _datapoint, float* _label, int _nSteps) 
{
	std::copy(_datapoint, _datapoint + datapointSize, &x[0][sizes[0]]);

#ifdef LABEL_IS_DATAPOINT
	int labelSize = sizes[0] - datapointSize;
	std::copy(_label, _label + labelSize, &x[0][datapointSize]);
#else
	std::copy(_label, _label + sizes[nLayers - 1], &x[0][nLayers - 1]);
	for (int j = 0; j < sizes[nLayers - 1]; j++)
	{
		fx[nLayers - 1][j] = tanhf(x[nLayers - 1][j]);
	}
#endif

	//setXtoMAP(false);
	sampleX(false);

	for (int i = 0; i < _nSteps; i++)
	{
		simultaneousAscentStep(true);
		LOG(computePerVariateEnergy() * 100.f << std::setprecision(3));
	}
	LOGL("\n"); 

	
	updateParameters();
}

void FCN::evaluate(float* _datapoint, int _nSteps)
{
	std::copy(_datapoint, _datapoint + datapointSize, &x[0][sizes[0]]);

	setXtoMAP(false);
	//sampleX(false);

	for (int i = 0; i < _nSteps; i++)
	{
		simultaneousAscentStep(false);
		LOG(computePerVariateEnergy() * 100.f << std::setprecision(3));
	}
	LOGL("\n");
}

