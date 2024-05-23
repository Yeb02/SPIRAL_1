#pragma once

#include "SPIRAL_includes.h"


struct FCN 
{
	int nLayers;
	int* sizes;

	int datapointSize;

	float weightRegularization; 
	float gradientStepSize; 

	
	// pre synaptic activations
	float** x;

	// post synaptic activations. To avoid bugs, whenever x is changed fx must also be updated.
	float** fx;

	// prediction errors. mus are not explicited in a variable since they are temp quantities
	float** epsilon;

	// the precision (inverse variance) of the activation
	float** tau;

#ifdef DYNAMIC_PRECISIONS
	
	TODO

#endif

	// the activation weights of the most likely model right now. 
	float** wx_variates;
	// the MAP mean of the gaussian distribution on independent activation weights
	float** wx_mean;
	// the MAP precision of the gaussian distribution on independent activation weights
	float** wx_precision;


	// the activation biases of the most likely model right now. 
	float** bx_variates;
	// the MAP mean of the gaussian distribution on independent activation biases
	float** bx_mean;
	// the MAP precision of the gaussian distribution on independent activation biases
	float** bx_precision;


#ifdef PROSPECTIVE_GRAD
	float internalGradientStepSize;
	int nInternalSteps;

	float** deltaX;
	
	float** F1;
	float** F2;
	float** F3;
	float** deltaMu;
#endif

	FCN(const int _nLayers, int* _sizes, int _datapointSize, float _weightRegularization, float _gradientStepSize);

	~FCN();

	void setXtoMAP(bool supervised);
	void sampleX(bool supervised);

	void setWBtoMAP(); 
	void sampleWB(); 

	void computeEpsilons(int layer);
	void computeAllEpsilons();

	void simultaneousAscentStep(bool supervised);

#ifdef PROSPECTIVE_GRAD
	void externalGradientStep(bool supervised);
	void internalGradientStep(bool supervised);
	void setOptimalWB();
	void initializeDeltaX(bool supervised);
#endif

	void updateParameters();

	float computePerActivationEnergy();
	float computePerVariateEnergy();


	void learn(float* datapoint, float* label, int nSteps);
	void evaluate(float* datapoint, int nSteps);

};