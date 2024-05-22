#include "FCN.h"

#include <iomanip> // std::setprecision


FCN::FCN(const int _nLayers, int* _sizes, int _datapointSize, float _weightRegularization, float _gradientStepSize) :
	nLayers(_nLayers), sizes(_sizes), gradientStepSize(_gradientStepSize), datapointSize(_datapointSize),
	weightRegularization(_weightRegularization)
{


	wx_variates = new float*[nLayers-1];
	wx_mean = new float*[nLayers-1];
	wx_precision = new float*[nLayers-1];
	wx_importance = new float*[nLayers-1];
	
	bx_variates = new float*[nLayers];
	bx_mean = new float*[nLayers];
	bx_precision = new float*[nLayers];
	bx_importance = new float*[nLayers];

	x = new float* [nLayers];
	fx = new float* [nLayers];
	epsilon = new float* [nLayers];
	tau = new float* [nLayers];

#ifdef DYNAMIC_PRECISIONS
	
#endif

	for (int i = 0; i < nLayers-1; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		float f = powf((float)sizes[i + 1], .5f);

		wx_variates[i] = new float[s];
		wx_mean[i] = new float[s];
		wx_precision[i] = new float[s];
		wx_importance[i] = new float[s];
		
		
		std::fill(wx_precision[i], wx_precision[i] + s, 1.0f); // TODO f or 1.f ?
		std::fill(wx_importance[i], wx_importance[i] + s, 1.0f); // TODO thats the prior's strength, give it a good think

#ifdef DYNAMIC_PRECISIONS

#endif

		for (int j = 0; j < s; j++) {
			wx_mean[i][j] = NORMAL_01 * wx_precision[i][j]; 
		}
	}




	for (int i = 0; i < nLayers; i++)
	{

		bx_variates[i] = new float[sizes[i]];
		bx_mean[i] = new float[sizes[i]];
		bx_precision[i] = new float[sizes[i]];
		bx_importance[i] = new float[sizes[i]];
		

		x[i] = new float[sizes[i]];
		fx[i] = new float[sizes[i]];
		epsilon[i] = new float[sizes[i]];
		tau[i] = new float[sizes[i]];

		std::fill(bx_precision[i], bx_precision[i] + sizes[i], 1.0f);
		std::fill(bx_importance[i], bx_importance[i] + sizes[i], 1.0f); // TODO thats the prior's strength, give it a good think

#ifdef DYNAMIC_PRECISIONS

#else
		std::fill(tau[i], tau[i] + sizes[i], 1.0f);
#endif

		for (int j = 0; j < sizes[i]; j++) {
			bx_mean[i][j] = NORMAL_01 * .3f;
		}

	}

	setWBtoMAP();
	//sampleWB();
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
		delete[] bx_importance[i];

		if (i == nLayers - 1) break;

		delete[] wx_variates[i];
		delete[] wx_mean[i];
		delete[] wx_precision[i];
		delete[] wx_importance[i];
	}
}




void FCN::sampleX(bool supervised)
{
#ifdef LABEL_IS_DATAPOINT
	for (int i = nLayers - 1; i >= 1; i--)
#else
	for (int i = nLayers - 1 - supervised; i >= 1; i--)
#endif
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

#ifdef LABEL_IS_DATAPOINT
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
#endif
}


void FCN::setXtoMAP(bool supervised)
{

#ifdef LABEL_IS_DATAPOINT
	for (int i = nLayers - 1; i >= 1; i--)
#else
	for (int i = nLayers - 1 - supervised; i >= 1; i--)
#endif
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

#ifdef LABEL_IS_DATAPOINT
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
#endif
}

void FCN::setWBtoMAP() 
{
	for (int i = 0; i < nLayers - 1; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		for (int j = 0; j < s; j++) {
			wx_variates[i][j] = wx_mean[i][j];
		}
	}

	for (int i = 0; i < nLayers; i++)
	{
		for (int j = 0; j < sizes[i]; j++) {
			bx_variates[i][j] = bx_mean[i][j];
		}
	}
}

void FCN::sampleWB() 
{
	for (int i = 0; i < nLayers - 1; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		for (int j = 0; j < s; j++) {
			wx_variates[i][j] = wx_mean[i][j] + NORMAL_01 / wx_precision[i][j];
		}
	}

	for (int i = 0; i < nLayers; i++)
	{
		for (int j = 0; j < sizes[i]; j++) {
			bx_variates[i][j] = bx_mean[i][j] + NORMAL_01 / bx_precision[i][j];
		}
	}
};


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
#ifdef LABEL_IS_DATAPOINT
		if (i == 0 && !supervised) 
		{
			for (int j = datapointSize; j < sizes[i]; j++) {
				float gradx = - epsilon[i][j] * tau[i][j];
				x[i][j] += gradientStepSize * gradx;
			}
		}
		else if (i > 0) {
#else
		if (i > 0 && i < nLayers - supervised) {
#endif
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
					float gradw = - weightRegularization * wx_variates[i - 1][id];
					gradw += -(wx_variates[i - 1][id] - wx_mean[i - 1][id]) * wx_precision[i - 1][id];
					gradw += epsilon[i-1][j] * tau[i-1][j] * fx[i][k];
					wx_variates[i-1][id] += gradientStepSize * gradw;
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
			float gradb = -(bx_variates[i][j] - bx_mean[i][j]) * bx_precision[i][j];
			gradb += epsilon[i][j] * tau[i][j];
			bx_variates[i][j] += gradb * gradientStepSize;
		}
	}
}



void FCN::updateParameters() 
{
	
	for (int i = 0; i < nLayers; i++)
	{
		// w[i]
		if (i < nLayers - 1)
		{
			int id = 0;
			for (int j = 0; j < sizes[i]; j++)
			{
				for (int k = 0; k < sizes[i + 1]; k++)
				{
					//float ew = wx_variates[i][id] - wx_mean[i][id];
					//float importanceWeight = powf(wx_precision[i][id], -.5f) * expf(.5f *ew* ew* wx_precision[i][id]);// * sqrt(2*pi) 
					//
					//float new_importance = wx_importance[i][id] + importanceWeight;
					//float new_mu = (wx_mean[i][id] * wx_importance[i][id] + importanceWeight * wx_variates[i][id]) / new_importance;
					//float delta_mu = new_mu - wx_mean[i][id];
					//float newSigma2 = wx_importance[i][id] * (1.0f / (wx_precision[i][id] * new_importance) + delta_mu * delta_mu / importanceWeight);

					//wx_mean[i][id] = new_mu;
					//wx_precision[i][id] = 1.0f / newSigma2;
					//wx_importance[i][id] = new_importance;
					
					wx_mean[i][id] = wx_variates[i][id];
					//float importanceWeight = 1.0f;
					float importanceWeight = .5f * tau[i][j] * powf(fx[i+1][k], 2.0f);
					wx_precision[i][id] += importanceWeight;
					
					id++;
				}
			}
		}

		// b[i]
		for (int j = 0; j < sizes[i]; j++)
		{
			//float eb = bx_variates[i][j] - bx_mean[i][j];
			//float importanceWeight = powf(bx_precision[i][j], -.5f) * expf(.5f * eb * eb * bx_precision[i][j]);// * sqrt(2*pi) 

			//float new_importance = bx_importance[i][j] + importanceWeight;
			//float new_mu = (bx_mean[i][j] * bx_importance[i][j] + importanceWeight * bx_variates[i][j]) / new_importance;
			//float delta_mu = new_mu - bx_mean[i][j];
			//float newSigma2 = bx_importance[i][j] * (1.0f / (bx_precision[i][j] * new_importance) + delta_mu * delta_mu / importanceWeight);

			//bx_mean[i][j] = new_mu;
			//bx_precision[i][j] = 1.0f / newSigma2;
			//bx_importance[i][j] = new_importance;

			bx_mean[i][j] = bx_variates[i][j];
			//float importanceWeight = 1.0f;
			float importanceWeight = .5f * tau[i][j];
			bx_precision[i][j] += importanceWeight;
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
		n += 2 * sizes[i];
		computeEpsilons(i);
		for (int j = 0; j < sizes[i]; j++)
		{
			E += .5f * epsilon[i][j] * epsilon[i][j] * tau[i][j]; // - logf(tau[i][j]); // + constants
			float eb = bx_variates[i][j] - bx_mean[i][j];
			E += .5f * eb * eb * bx_precision[i][j]; // - logf(bx_precision[i][j]); // + constants
			
		}

		if (i == nLayers - 1) break;

		float s = sizes[i] * sizes[i + 1];
		n += s;
		for (int j = 0; j < s; j++)
		{
			float ew = wx_variates[i][j] - wx_mean[i][j];
			E += .5f * ew * ew * wx_precision[i][j]; // -logf(wx_precision[i][j]); // + constants
		}
	}

	return E / (float)n;
}


void FCN::learn(float* _datapoint, float* _label, int _nSteps) 
{
	std::copy(_datapoint, _datapoint + datapointSize, x[0]);

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

	//setXtoMAP(true);
	sampleX(true);

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
	std::copy(_datapoint, _datapoint + datapointSize, x[0]);


	setXtoMAP(false);
	//sampleX(false);

	for (int i = 0; i < _nSteps; i++)
	{
		simultaneousAscentStep(false);
		LOG(computePerVariateEnergy() * 100.f << std::setprecision(3));
	}
	LOGL("\n");
}

