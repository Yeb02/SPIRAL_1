#include "FCN.h"

#include <iomanip> // std::setprecision


FCN::FCN(const int _nLayers, int* _sizes, int _datapointSize) :
	nLayers(_nLayers), sizes(_sizes), datapointSize(_datapointSize)
{
	wReg = 1.f;  
	xReg = 1.f;  
	xlr = .5f;
	wlr = .2f;
	certaintyDecay = .2f;

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


	float priorImportance = 1.f;

	for (int i = 0; i < nLayers; i++)
	{
		int s = sizes[i] * sizes[i + 1];
		

		wx_variates[i] = new float[s];
		wx_mean[i] = new float[s];
		wx_precision[i] = new float[s];
		
		
		 // TODO He or 1.0f ? Thats the prior's strength, give it a good think
		float fw = powf((float)(sizes[i]), .5f);
		std::fill(wx_precision[i], wx_precision[i] + s, priorImportance * fw);

#ifdef DYNAMIC_PRECISIONS

#endif

		for (int j = 0; j < s; j++) {
			wx_mean[i][j] = NORMAL_01 * .01f; 
		}
	

		bx_variates[i] = new float[sizes[i]];
		bx_mean[i] = new float[sizes[i]];
		bx_precision[i] = new float[sizes[i]];
		

		x[i] = new float[sizes[i]];
		fx[i] = new float[sizes[i]];
		epsilon[i] = new float[sizes[i]];
		tau[i] = new float[sizes[i]];


#ifdef LABEL_IS_DATAPOINT
		if (i == 0) {
#else
		if ((i == 0) || (i == nLayers - 1) ) {
#endif
			// negligible strength because our prior is meaningless compared to the true observed data. 
			// Unless very small, extremely noisy sample...
			std::fill(bx_precision[i], bx_precision[i] + sizes[i], .001f); 
		}
		else {
			float fb = powf((float)(sizes[i]), .5f);
			std::fill(bx_precision[i], bx_precision[i] + sizes[i], priorImportance * fb); // TODO thats the prior's strength, give it a good think
		}
		

		std::fill(tau[i], tau[i] + sizes[i], 1.0f); 
#ifdef DYNAMIC_PRECISIONS

#else
		std::fill(tau[i], tau[i] + sizes[i], 1.0f);
#endif

		for (int j = 0; j < sizes[i]; j++) {
			bx_mean[i][j] = NORMAL_01 * .01f; 
		}


	}

	setWBtoMAP();
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

		delete[] wx_variates[i];
		delete[] wx_mean[i];
		delete[] wx_precision[i];

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
				float gradx = -epsilon[i][j]; // if no nat grads *tau[i][j];
				x[i][j] += xlr * gradx; 
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
					grad_acc += epsilon[i - 1][k] * tau[i - 1][k] * wx_variates[i - 1][k*sizes[i] + j]; // TODO remove *tau[i - 1][k] if using nat grads ???
				}
				gradx += fprime * grad_acc; // / sqrtf((float)sizes[i - 1]); // TODO 
				// TODO . Est ce grave ? Je dirai que non comme il n'y a pas de second ordre donc d'interactions, et que les valeurs relatives du gradient
				// d'un x à l'autre n'importent donc pas. Meilleures perfs avec clamp, mais convergence plus rapide sans clamp.
				x[i][j] += std::clamp(xlr * gradx / tau[i][j], -.3f, .3f); // remove /tau[i][j]; if no nat grads 
				
				// fx updated further down
			}
		}
		


		// wl-1, bl then f(xl)
		//if (supervised && false) // TODO
		//{
		//	int id = 0;
		//	for (int j = 0; j < sizes[i-1]; j++)
		//	{
		//		for (int k = 0; k < sizes[i]; k++)
		//		{
		//			float gradw = - weightRegularization * wx_variates[i - 1][id];
		//			gradw += -(wx_variates[i - 1][id] - wx_mean[i - 1][id]) * wx_precision[i - 1][id];
		//			gradw += epsilon[i-1][j] * tau[i-1][j] * fx[i][k];
		//			wx_variates[i-1][id] += gradientStepSize * gradw / wx_precision[i - 1][id];
		//			id++;

		//		}
		//	}

		//	for (int j = 0; j < sizes[i]; j++)
		//	{
		//		float gradb = -(bx_variates[i][j] - bx_mean[i][j]) * bx_precision[i][j];
		//		gradb += epsilon[i][j] * tau[i][j];
		//		bx_variates[i][j] += gradb * gradientStepSize / bx_precision[i][j];
		//	}
		//}
		if (i > 0) 
		{
			for (int j = 0; j < sizes[i]; j++) {
				fx[i][j] = tanhf(x[i][j]);
			}
		}
	}
}


void FCN::updateParameters() 
{
	computeAllEpsilons();
	float importanceLearningRate = 1.f;

	for (int i = 0; i < nLayers; i++)
	{
		
		int id = 0;
		

		for (int j = 0; j < sizes[i]; j++)
		{
			for (int k = 0; k < sizes[i + 1]; k++)
			{
				
				//  * tau[i][j] * powf(fx[i + 1][k], 2.0f); does not work. TODO * tau[i][j] only ? When dynamic taus.
				wx_precision[i][id] += importanceLearningRate;
				wx_precision[i][id] *= 1.0f - certaintyDecay; // TODO heuristics here too ?
				
				wx_mean[i][id] = wx_variates[i][id] * wReg;
				
				id++;
			}

			bx_mean[i][j] = bx_variates[i][j];

			bx_precision[i][j] += importanceLearningRate; // TODO * tau[i][j] ? When dynamic taus.
			bx_precision[i][j] *= 1.0f - certaintyDecay; // TODO heuristics here too ? 
		}
	}
}

float FCN::computePerActivationEnergy()
{
	float E = 0.f;
	int n = 0;

	for (int i = 0; i < nLayers; i++)
	//for (int i = 2; i < 3; i++)

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

		int s = sizes[i] * sizes[i + 1];
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

	setXtoMAP(true); // after quick and noisy tests, seems a bit more efficient. Can also use neither, and keep previous activations.
	//sampleX(true);

	//float previousEnergy = computePerActivationEnergy();
	for (int i = 0; i < _nSteps; i++)
	{

		//LOG(previousEnergy * 100.f);
		simultaneousAscentStep(true);
		//float currentEnergy = computePerActivationEnergy();

		// Makes learning very unstable in some setting. Sometimes somewhat better results ! TODO theoretical study.
		// Moreover, should be a local operation since we are headed towards distributed computing.
		//if (currentEnergy / previousEnergy > .99f) {
			//setOptimalWB();
		//	LOG("WBU");
		//	currentEnergy = computePerActivationEnergy();
		//}
		
		//previousEnergy = currentEnergy;
	}
	//LOG(previousEnergy * 100.f);
	setOptimalWB();
	//LOG(" WBU  " << computePerActivationEnergy() * 100.f);


	//LOGL("\n"); 

	updateParameters();
}

void FCN::evaluate(float* _datapoint, int _nSteps)
{
	std::copy(_datapoint, _datapoint + datapointSize, x[0]);

	setWBtoMAP();

	setXtoMAP(false);
	//sampleX(false);

	for (int i = 0; i < _nSteps; i++)
	{
		//LOG(computePerActivationEnergy() * 100.f);
		simultaneousAscentStep(false);
	}
}


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

