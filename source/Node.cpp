#include "Node.h"

// Recommended values:
float Node::priorStrength = .2f;
float Node::activationDescentStepSize = .1f;
float Node::wxDescentStepSize = .1f;
float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .95f;
float Node::weightRegularization = .01f;
float Node::potentialConservation = .8f;


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
	epsilon = x - bx_variate;
	fx = tanhf(x);

#ifndef DYNAMIC_PRECISIONS
	tau = 1.0f;
#endif

	potential = epsilon * tau * epsilon * tau;
	accumulatedEnergy = .0f;
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


void Node::asynchronousWeightGradientStep()
{
	
	for (int k = 0; k < nParents; k++)
	{
		float pfx = parents[k]->fx;
		float delta = wxDescentStepSize * (epsilon * pfx * tau - (wx_variates[k] - wx_means[k]) * wx_precisions[k]) /
			(pfx * pfx * tau + wx_precisions[k]);
		wx_variates[k] += delta;
		epsilon -= delta * pfx;
	}
	float delta = wxDescentStepSize * (epsilon * tau - (bx_variate - bx_mean) * bx_precision) / (tau + bx_precision);
	bx_variate += delta;
	epsilon -= delta;
}


void Node::asynchronousActivationGradientStep()
{
	float grad = -epsilon * tau;
	float fprime = 1.0f - powf(fx, 2.f);
	float grad_acc = .0f;

	float H = tau;
	
	for (int k = 0; k < nChildren; k++) 
	{
		Node& c = *children[k];
		float w = c.wx_variates[inChildID[k]];
		grad_acc += c.epsilon * c.tau * w;  

		float fpw = w * fprime;
		H += abs(c.tau * fpw * (c.epsilon * fx + fpw)); 
		//H += c.tau * fpw * (c.epsilon * fx + fpw);
		//H = std::max(H, abs(c.tau * fpw))
		// H = H; // just a reminder that simply setting H to tau has to be tested.
	}
	grad += fprime * grad_acc;

	potential = potential * potentialConservation + grad * grad * (1.f - potentialConservation);// TODO abs(grad) ?

	float oldX = x;

	// TODO when H is the exact second derivative, compare with and without clamping, both performance and convergence speed.
	//x += std::clamp(activationDescentStepSize * grad / H, -.2f, .2f); 
	x += activationDescentStepSize * grad / H; 
	//x += activationDescentStepSize * grad / std::min(abs(H), activationDescentStepSize*.1f); 
	
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

		wx_means[k] = wx_variates[k] * (1.0f - weightRegularization * wx_variates[k]); // TODO regularization should be in the update function, not the learn one
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

void Node::setActivation(float newX)
{
	epsilon = epsilon + newX - x;
	x = newX;

	float new_fx = tanhf(x);
	for (int k = 0; k < nChildren; k++)
	{
		children[k]->epsilon += (new_fx - fx) * children[k]->wx_variates[inChildID[k]];
	}
	fx = new_fx;
}