#pragma once

#include "SPIRAL_includes.h"


struct FCN 
{
	int nLayers;
	int* sizes;
	int datapointSize;


	//Must be set manually after construction:

	float wReg, xReg; // regularization strength 
	float xlr, wlr;   // learning rates aka gradient step sizes 
	float certaintyDecay;


	
	// pre synaptic activations
	float** x;

	// post synaptic activations. To avoid bugs, whenever x is changed fx must also be updated.
	float** fx;

	// prediction errors. mus are not explicited in a variable since they are temp quantities
	float** epsilon;

	// the precision (inverse variance) of the activation
	float** tau;

#ifdef DYNAMIC_PRECISIONS
	
	// TODO

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



	FCN(const int _nLayers, int* _sizes, int _datapointSize);

	~FCN();

	void setXtoMAP(bool supervised);
	void sampleX(bool supervised);

	void setWBtoMAP(); 
	void sampleWB(); 

	void computeEpsilons(int layer);
	void computeAllEpsilons();

	void simultaneousAscentStep(bool supervised);
	void normalizedAscentStep(bool supervised);

	void setOptimalWB();
	void gradStepWB();


	void updateParameters();

	float computePerActivationEnergy();
	float computePerVariateEnergy();


	void learn(float* datapoint, float* label, int nSteps);
	void evaluate(float* datapoint, int nSteps);

};