#include "Node.h"

// Recommended values:
float Node::priorStrength = .2f;
float Node::activationDescentStepSize = .1f;
float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .95f;
float Node::weightRegularization = 1.0f;


Node::Node(int _nChildren, Node** _children) :
	nChildren(_nChildren), children(_children)
{
	nParents = 0;
	inChildID = nullptr;
	parents = nullptr;
	parentArraySize = 0;
	
	bx_mean = .0f;
	bx_precision = priorStrength;
	bx_variate = bx_mean;

	wx_variates = nullptr;
	wx_means = nullptr;
	wx_precisions = nullptr;

	x = .0f; 
	fx = .0f; 
	epsilon = .0f;

#ifndef DYNAMIC_PRECISIONS
	tau = 1.0f;
#endif
}

Node::~Node()
{
	delete[] children;
	delete[] parents;
	delete[] inChildID;
	delete[] wx_variates;
	delete[] wx_means;
	delete[] wx_precisions;
}


void Node::asynchronousActivationGradientStep()
{
	float grad = -epsilon * tau;
	float fprime = 1.0f - powf(fx, 2.f);
	float grad_acc = .0f;

	float H = tau;
	float H_acc = .0f;
	
	for (int k = 0; k < nChildren; k++) 
	{
		Node& c = *children[k];
		float w = c.wx_variates[inChildID[k]];
		grad_acc += c.epsilon * c.tau * w;  

		float fw = w * fprime;
		H_acc += c.tau * fw * (c.epsilon * fx + fw);
	}
	grad += fprime * grad_acc;
	H += grad_acc;


	float oldX = x;
	//x += std::clamp(activationDescentStepSize * grad / H, -.2f, .2f); 
	x += activationDescentStepSize * grad / H; // TODO compare with and without clamping, both performance and convergence speed.
	
	epsilon = epsilon + x - oldX;
	float new_fx = tanhf(x);
	for (int k = 0; k < nChildren; k++)
	{
		children[k]->epsilon += (new_fx - fx) * children[k]->wx_variates[inChildID[k]];
	}
	fx = new_fx;
}


void Node::updateIncomingXWBvariates()
{
	
	float a1 = bx_mean;
	float a2 = 1.0f / bx_precision;

	for (int k = 0; k < nParents; k++)
	{
		a1 += parents[k]->fx * wx_means[k];
		a2 += parents[k]->fx * parents[k]->fx / wx_precisions[k];
	}
	a2 *= tau;

	epsilon = (x - a1) / (1.0f + a2);
	float fcom = tau * epsilon;
		
	for (int k = 0; k < nParents; k++)
	{
		wx_variates[k] = wx_means[k] + fcom * parents[k]->fx / wx_precisions[k];
	}
	bx_variate = bx_mean + fcom / bx_precision;

	
}

void Node::learnIncomingXWBvariates()
{
	for (int k = 0; k < nParents; k++)
	{
		// * .5f * tau[i][j] * powf(fx[i + 1][k], 2.0f); does not work surprisingly. TODO * tau only ? When dynamic taus.
		wx_precisions[k] += observationImportance;
		wx_precisions[k] *= certaintyDecay; // TODO heuristics here too ?

		wx_means[k] = wx_variates[k] * weightRegularization;
	}
	
	bx_mean = bx_variate;

	bx_precision += observationImportance; // TODO * tau[i][j] ? When dynamic taus.
	bx_precision *= certaintyDecay; // TODO heuristics here too ? 
}

void Node::computeEpsilon() 
{
	epsilon = bx_variate;
	for (int i = 0; i < nParents; i++) 
	{
		epsilon += wx_variates[i] * parents[i]->fx;
	}
	epsilon = x - epsilon;
}