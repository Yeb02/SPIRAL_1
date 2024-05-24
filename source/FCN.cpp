#include "FCN.h"

#include <iomanip> // std::setprecision


FCN::FCN(const int _nLayers, int* _sizes, int _datapointSize, float _weightRegularization, float _gradientStepSize) :
	nLayers(_nLayers), sizes(_sizes), gradientStepSize(_gradientStepSize), datapointSize(_datapointSize),
	weightRegularization(_weightRegularization)
{


	wx_variates = new float*[nLayers];
	wx_mean = new float*[nLayers];
	wx_precision = new float*[nLayers];
	
	bx_variates = new float*[nLayers];
	bx_mean = new float*[nLayers];
	bx_precision = new float*[nLayers];

	x = new float* [nLayers];
	fx = new float* [nLayers];
	epsilon = new float* [nLayers];
	tau = new float* [nLayers];


#ifdef PROSPECTIVE_GRAD
	internalGradientStepSize = .0f;
	nInternalSteps = 0;


	deltaX = new float* [nLayers];
	F1 = new float* [nLayers];
	F2 = new float* [nLayers];
	F3 = new float* [nLayers];
	deltaMu = new float* [nLayers];
#endif

#ifdef BARYGRAD
	deltaX = new float* [nLayers];
#endif


#ifdef DYNAMIC_PRECISIONS
	
#endif

	for (int i = 0; i < nLayers; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		float He = powf((float)s, .5f);

		wx_variates[i] = new float[s];
		wx_mean[i] = new float[s];
		wx_precision[i] = new float[s];
		
		
		//std::fill(wx_precision[i], wx_precision[i] + s, He); // TODO He or 1.0f ? Thats the prior's strength, give it a good think
		std::fill(wx_precision[i], wx_precision[i] + s, 1.0f); // TODO He or 1.0f ? Thats the prior's strength, give it a good think

#ifdef DYNAMIC_PRECISIONS

#endif

		for (int j = 0; j < s; j++) {
			//wx_mean[i][j] = NORMAL_01 / wx_precision[i][j]; 
			wx_mean[i][j] = NORMAL_01 / He; 
		}
	

		bx_variates[i] = new float[sizes[i]];
		bx_mean[i] = new float[sizes[i]];
		bx_precision[i] = new float[sizes[i]];
		

		x[i] = new float[sizes[i]];
		fx[i] = new float[sizes[i]];
		epsilon[i] = new float[sizes[i]];
		tau[i] = new float[sizes[i]];

		std::fill(bx_precision[i], bx_precision[i] + sizes[i], 1.0f); // TODO thats the prior's strength, give it a good think

#ifdef DYNAMIC_PRECISIONS

#else
		std::fill(tau[i], tau[i] + sizes[i], 1.0f);
#endif

		for (int j = 0; j < sizes[i]; j++) {
			bx_mean[i][j] = NORMAL_01 * .3f;
		}

#ifdef PROSPECTIVE_GRAD
		deltaX[i] = new float [sizes[i]];
		F1[i] = new float [sizes[i]];
		F2[i] = new float [sizes[i]];
		F3[i] = new float [sizes[i]];
		deltaMu[i] = new float [sizes[i]];
#endif

#ifdef BARYGRAD
		deltaX[i] = new float[sizes[i]];
#endif

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

		delete[] wx_variates[i];
		delete[] wx_mean[i];
		delete[] wx_precision[i];

#ifdef PROSPECTIVE_GRAD
		delete[] deltaX[i];
		delete[] F1[i];
		delete[] F2[i];
		delete[] F3[i];
		delete[] deltaMu[i];
#endif

#ifdef BARYGRAD
		delete[] deltaX[i];
#endif
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

		int id = 0;
		for (int j = 0; j < sizes[i]; j++)
		{
			x[i][j] = bx_variates[i][j];

			for (int k = 0; k < sizes[i + 1]; k++)
			{
				x[i][j] += wx_variates[i][id] * fx[i + 1][k];
				id++;
			}

			fx[i][j] = tanhf(x[i][j]);
		}
	}

#ifdef LABEL_IS_DATAPOINT
	if (!supervised) {
		int id = 0;
		for (int j = datapointSize; j < sizes[0]; j++)
		{
			x[0][j] = bx_variates[0][j];
			for (int k = 0; k < sizes[1]; k++)
			{
				x[0][j] += wx_variates[0][id] * fx[1][k];
				id++;
			}
			fx[0][j] = tanhf(x[0][j]);
		}
	}
#endif
}


void FCN::setWBtoMAP() 
{
	for (int i = 0; i < nLayers; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		for (int j = 0; j < s; j++) {
			wx_variates[i][j] = wx_mean[i][j];
		}
		
		for (int j = 0; j < sizes[i]; j++) {
			bx_variates[i][j] = bx_mean[i][j];
		}
	}
}

void FCN::sampleWB() 
{
	for (int i = 0; i < nLayers; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		for (int j = 0; j < s; j++) {
			wx_variates[i][j] = wx_mean[i][j] + NORMAL_01 / wx_precision[i][j];
		}

		for (int j = 0; j < sizes[i]; j++) {
			bx_variates[i][j] = bx_mean[i][j] + NORMAL_01 / bx_precision[i][j];
		}
	}
};


void FCN::computeEpsilons(int l)
{

	int id = 0;
	for (int j = 0; j < sizes[l]; j++)
	{
		epsilon[l][j] = bx_variates[l][j];
		for (int k = 0; k < sizes[l + 1]; k++)
		{
			epsilon[l][j] += wx_variates[l][id] * fx[l + 1][k];
			id++;
		}
		epsilon[l][j] = x[l][j] - epsilon[l][j];
	}
}

void FCN::computeAllEpsilons() 
{
	for (int l = 0; l < nLayers; l++)
	{
		computeEpsilons(l);
	}
}


void FCN::simultaneousAscentStep(bool supervised)
{
	computeAllEpsilons();

	
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
		


		// wl-1, bl then f(xl)
		if (supervised)
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

			for (int j = 0; j < sizes[i]; j++)
			{
				float gradb = -(bx_variates[i][j] - bx_mean[i][j]) * bx_precision[i][j];
				gradb += epsilon[i][j] * tau[i][j];
				bx_variates[i][j] += gradb * gradientStepSize;
			}
		}
		if (i > 0) 
		{
			for (int j = 0; j < sizes[i]; j++) {
				fx[i][j] = tanhf(x[i][j]);
			}
		}

	}
}

void FCN::normalizedAscentStep(bool supervised) 
{
	computeAllEpsilons();

	for (int i = 0; i < nLayers; i++) {
		int id = 0;
		for (int j = 0; j < sizes[i]; j++) {
			deltaX[i][j] = 1.0f/tau[i][j];
			for (int k = 0; k < sizes[i+1]; k++) {
				deltaX[i][j] += powf(wx_variates[i][id] * (1.0f - fx[i + 1][k] * fx[i + 1][k]), 2.0f) / tau[i + 1][k];
				id++;
			}
			deltaX[i][j] = epsilon[i][j] / deltaX[i][j];
		}
	}

	for (int i = 1; i < nLayers-supervised; i++) {
		for (int j = 0; j < sizes[i]; j++) {
			float num = .0f;
			float den = .0f;

			for (int k = 0; k < sizes[i - 1]; k++) {
				num += tau[i - 1][k] * deltaX[i - 1][k] * wx_variates[i-1][k*sizes[i] + j];
				den += tau[i - 1][k];
			}
			num *= (1.0f - fx[i][j] * fx[i][j]) / tau[i][j];

			num += deltaX[i][j]; // * tau[i][j] / tau[i][j]
			den += tau[i][j];

			x[i][j] += gradientStepSize * num / den;
			fx[i][j] = tanhf(x[i][j]);
		}
	}
}


void FCN::updateParameters() 
{
	
	for (int i = 0; i < nLayers; i++)
	{
		
		int id = 0;
		for (int j = 0; j < sizes[i]; j++)
		{
			for (int k = 0; k < sizes[i + 1]; k++)
			{
				wx_mean[i][id] = wx_variates[i][id];
				float importanceWeight = 1.f;
				//float importanceWeight = .5f * tau[i][j] * powf(fx[i+1][k], 2.0f) * .2f;
				wx_precision[i][id] += importanceWeight;
					
				id++;
			}
		}
		
		// b[i]
		for (int j = 0; j < sizes[i]; j++)
		{
			bx_mean[i][j] = bx_variates[i][j];
			float importanceWeight = 1.f;
			//float importanceWeight = .5f * tau[i][j] * .2f;
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
	std::copy(_label, _label + sizes[nLayers - 1], x[nLayers - 1]);
	for (int j = 0; j < sizes[nLayers - 1]; j++)
	{
		fx[nLayers - 1][j] = tanhf(x[nLayers - 1][j]);
	}
#endif

	setXtoMAP(true);
	//sampleX(true);

#ifdef PROSPECTIVE_GRAD
	initializeDeltaX(true);
#endif
	for (int i = 0; i < _nSteps; i++)
	{

#ifdef PROSPECTIVE_GRAD
		computeAllEpsilons();
		for (int j = 0; j < nInternalSteps; j++)
		{
			internalGradientStep(true);
		}
		externalGradientStep(true);
		setOptimalWB();
#endif 
		
#ifdef ORDINARY_GD
		simultaneousAscentStep(true);
		setOptimalWB();
#endif

#ifdef BARYGRAD
		normalizedAscentStep(true);
		
#endif

		LOG(computePerActivationEnergy() * 100.f << std::setprecision(3));
	}
	LOGL("\n"); 

	setOptimalWB();
	updateParameters();
}

void FCN::evaluate(float* _datapoint, int _nSteps)
{
	std::copy(_datapoint, _datapoint + datapointSize, x[0]);


	setXtoMAP(false);
	//sampleX(false);

#ifdef PROSPECTIVE_GRAD
	initializeDeltaX(false);
#endif

	for (int i = 0; i < _nSteps; i++)
	{
#ifdef PROSPECTIVE_GRAD
		computeAllEpsilons();
		for (int j = 0; j < nInternalSteps; j++)
		{
			internalGradientStep(false);
		}
		externalGradientStep(false);
#endif 

#ifdef ORDINARY_GD
		simultaneousAscentStep(false);
#endif

#ifdef BARYGRAD
		normalizedAscentStep(false);
#endif

		LOG(computePerActivationEnergy() * 100.f << std::setprecision(3));
	}
}


#ifdef PROSPECTIVE_GRAD
void FCN::externalGradientStep(bool supervised)
{
#ifdef LABEL_IS_DATAPOINT
	for (int i = nLayers - 1; i >= 1; i--)
#else
	for (int i = nLayers - 1 - supervised; i >= 1; i--)
#endif
	{
		for (int j = 0; j < sizes[i]; j++)
		{
			x[i][j] += deltaX[i][j];
			fx[i][j] = tanhf(x[i][j]);
		}
	}

#ifdef LABEL_IS_DATAPOINT
	if (!supervised) {
		int id = 0;
		for (int j = datapointSize; j < sizes[0]; j++)
		{
			x[0][j] += deltaX[0][j];
			fx[0][j] = tanhf(x[0][j]);
		}
	}
#endif
}

void FCN::internalGradientStep(bool supervised)
{

	// F1, F2
	for (int i = 0; i < nLayers; i++) 
	{
		
		for (int j = 0; j < sizes[i]; j++)
		{
			float fp = 1.0f - fx[i][j] * fx[i][j];
			F1[i][j] = fp * deltaX[i][j] * (1.0f - deltaX[i][j] * fx[i][j]);
			F2[i][j] = fp * (1.0f - 2.0f * deltaX[i][j] * fx[i][j]);
		}
	}

	// delta_mu, F3
	for (int i = 0; i < nLayers; i++)
	{
		int id = 0;
		for (int j = 0; j < sizes[i]; j++)
		{
			deltaMu[i][j] = .0f;
			for (int k = 0; k < sizes[i+1]; k++)
			{
				deltaMu[i][j] += F1[i+1][k] * wx_variates[i][id];
				id++;
			}
			F3[i][j] = tau[i][j] * (deltaMu[i][j] - epsilon[i][j] - deltaX[i][j]);
		}
	}


	// gradient, update deltaX
	for (int i = 0; i < nLayers; i++)
	{
		float lambda_xl = 1.0f / ((float)sizes[i-1] + 1.0f); // more generally, should be nChildren + 1 
		float lambda_mulm1 = 1.0f / (float)sizes[i]; // more generally, should be nParents and therefore a per node factor.


		for (int j = 0; j < sizes[i]; j++)
		{
			float a1 = .0f, a2 = .0f;
			for (int k = 0; k < sizes[i - 1]; k++)
			{
				a1 += F3[i-1][k] * wx_variates[i-1][k * sizes[i] + j];
				a2 += tau[i-1][k] * deltaMu[i-1][k] * wx_variates[i-1][k * sizes[i] + j];
			}

			float grad = -F3[i][j] * gradientStepSize + F2[i][j] * (a1 * gradientStepSize + a2 * lambda_mulm1) + tau[i][j] * deltaX[i][j] * lambda_xl;
			deltaX[i][j] += -internalGradientStepSize * grad;
		}
	}


	// Yeah, laziest implementation ever
	std::fill(deltaX[0], deltaX[0] + datapointSize, .0f);
	if (supervised) 
	{
#ifdef LABEL_IS_DATAPOINT
		std::fill(deltaX[0] + datapointSize, deltaX[0] + sizes[0], .0f);
#else
		int i = nLayers - 1;
		std::fill(deltaX[i], deltaX[i] + sizes[i], .0f);
#endif
	}

}



void FCN::initializeDeltaX(bool supervised) 
{
//#ifdef LABEL_IS_DATAPOINT
//	for (int i = nLayers - 1; i >= 1; i--)
//#else
//	for (int i = nLayers - 1 - supervised; i >= 1; i--)
//#endif
//	{
//		for (int j = 0; j < sizes[i]; j++)
//		{
//			deltaX[i][j] = .0f;
//		}
//	}
//
//#ifdef LABEL_IS_DATAPOINT
//	if (!supervised) {
//		int id = 0;
//		for (int j = datapointSize; j < sizes[0]; j++)
//		{
//			deltaX[0][j] = .0f;
//		}
//	}
//#endif

	for (int i = 0; i < nLayers; i++)
	{
		for (int j = 0; j < sizes[i]; j++)
		{
			deltaX[i][j] = .0f;
		}
	}
}

#endif

void FCN::setOptimalWB()
{
	for (int i = 0; i < nLayers; i++)
	{
		int id = 0;
		for (int j = 0; j < sizes[i]; j++)
		{
			float a1 = bx_mean[i][j];
			float a2 = 1.0f / bx_precision[i][j];

			for (int k = 0; k < sizes[i + 1]; k++)
			{
				a1 += fx[i + 1][k] * wx_mean[i][id];
				a2 += fx[i + 1][k] * fx[i + 1][k] / wx_precision[i][id];
				id++;
			}
			a2 *= tau[i][j];
			float fcom = tau[i][j] * (x[i][j] - a1) / (1.0f + a2);
			id -= sizes[i + 1];
			for (int k = 0; k < sizes[i + 1]; k++)
			{
				wx_variates[i][id] = wx_mean[i][id] + fcom * fx[i + 1][k] / wx_precision[i][id];
				id++;
			}
			bx_variates[i][j] = bx_mean[i][j] + fcom / bx_precision[i][j];
		}
	}
}

