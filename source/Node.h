#pragma once

#include "SPIRAL_includes.h"

// Be careful as epsilons are never completely recomputed and need be changed everytime any quantity involved in their
// formula is updated: the bias/activation, or the parents weigths/activations. Same for dynamic taus.
class Node 
{
public:

	static float xlr;
	
	static float xReg; 
	static float wxReg;
	static float wtReg;


	static float wxPriorStrength;
	static float wtPriorStrength;

	static float observationImportance;
	static float certaintyDecay;


	static float energyDecay;
	static float connexionEnergyThreshold;


	std::vector<Node*> children;
	std::vector<Node*> parents;
	std::vector<int> inParentsListIDs;

	float bx_variate;
	float bx_mean; 
	float bx_precision;
	
	// outgoing weights predicting the children's activations
	std::vector<float> wx_variates;
	std::vector<float> wx_means;
	std::vector<float> wx_precisions;

#ifdef DYNAMIC_PRECISIONS
	float bt_variate;
	float bt_mean;
	float bt_precision;

	// outgoing weights computing the children's t
	std::vector<float> wt_variates;
	std::vector<float> wt_means;
	std::vector<float> wt_precisions;

	// the prediction accumulator for tau = expf(t)
	float t;

#endif

	float x, fx, tau, epsilon, mu;

#if defined(DYNAMIC_PRECISIONS) || defined(FIXED_PRECISIONS_BUT_CONTRIBUTE)
	float leps;
	void computeLeps(); // to be called after activations convergence, before optimal weights computation and calcification
#endif

#ifdef FIXED_PRECISIONS_BUT_CONTRIBUTE
	float tau_mean, tau_precision;
#endif

	// for topology
	float accumulatedEnergy; 
	std::vector<float> connexionEnergies;


	Node(int _nChildren, Node** _children);

	~Node() {};


	void XGradientStep();



	void setAnalyticalWX();

	void setAnalyticalWT();

	

	// Sets the MAP ('mean') to the variate, and updates the precision. Also updates the energies ahead of topological operations.
	void calcifyWB();



	void pruneUnusedConnexions();

	// For benchmarking purposes
	void predictiveCodingWxGradientStep();

	// Updates the children's predicted quantities (mu and t) as well, which requires them to be up to date 
	// relative to the current parameters ! 
	// And a call to computeLocalQuantities must be performed afterwards by the children !
	void setActivation(float newX);


	// sets up (i.e. intitializes accumulators with the biases) this node to receive and 
	// make sense of prediction information regarding its mu and t
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// Sets all values depending directly on the raw predictions that the parent sent:
	// epsilon, tau.
	void computeLocalQuantities();

};