#pragma once

#include "SPIRAL_includes.h"
#include "Group.h"


class Node 
{
public:

	static float xlr;
	
	static float xReg; 
	static float wxReg;

	static float wxPriorStrength;

	static float observationImportance;
	static float certaintyDecay;


	float localXReg; // set at 0 for observation/label nodes by the parent network, xReg otherwise

#ifdef FREE_NODES
	bool isFree; // set to true for nodes that must be inferred and do not have children. Typically the label.
#endif

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


	float x, fx, epsilon, mu;


	Group* group;



	Node(Group* _group);

	~Node() {};



	void addChildren(Node** newChildren, int nNewChildren, int specialCase);

	void addParents(Node** newParents, int* newInParentIDs, int nNewParents);




	void XGradientStep();

	void analyticalXUpdate();




	void setAnalyticalWX();

	void calcifyWB();


	// For benchmarking purposes, the original predictive coding algorithm can be used. See config.h 
	void predictiveCodingWxGradientStep();



	// Updates this node's x, eps, fx, and propagates the new fx to the children's mu. Does not update the children's epsilon !
	// Used only by the parent Network when clamping to obseration values.
	void setActivation(float newX);

	// intitializes mu to the bias, to then receive w*fx from the parents 
	void prepareToReceivePredictions();

	// sends w*fx to the children for their mus
	void transmitPredictions();
};