#include "Node.h"

// Recommended values:
float Node::priorStrength = .2f;
float Node::xlr = 1.f;
float Node::wxlr = .1f;
float Node::wtlr = .1f;
float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .95f;
float Node::xReg = .01f;
float Node::wxReg = .01f;
float Node::wtReg = .01f;


Node::Node(int _nChildren, Node** _children) :
	nChildren(_nChildren), children(_children)
{
	
	bx_mean = NORMAL_01 * .01f;
	bx_precision = priorStrength;
	bx_variate = bx_mean;

	wx_means = new float[nChildren];
	wx_precisions = new float[nChildren];
	wx_variates = new float[nChildren];

	std::fill(wx_precisions, wx_precisions + nChildren, priorStrength);
	for (int i = 0; i < nChildren; i++) wx_means[i] = NORMAL_01 * .01f;
	std::copy(wx_means, wx_means + nChildren, wx_variates);

	mu = bx_variate;
	x = NORMAL_01 * .01f;
	epsilon = x - bx_variate;
	fx = tanhf(x);

#ifndef DYNAMIC_PRECISIONS
	tau = 1.0f;
#endif

}

Node::~Node()
{
	delete[] children;
	delete[] wx_variates;
	delete[] wx_means;
	delete[] wx_precisions;
}



void Node::asynchronousGradientStep()
{
	float grad = -epsilon * tau;
	float fprime = 1.0f - powf(fx, 2.f);
	float grad_acc = .0f;

	float H = tau;
	
	for (int k = 0; k < nChildren; k++) 
	{
		Node& c = *children[k];
		grad_acc += c.epsilon * c.tau * wx_variates[k];

		float fpw = wx_variates[k] * fprime;
		H += abs(c.tau * fpw * (c.epsilon * fx + fpw));  // L1 H
		//H += c.tau * fpw * (c.epsilon * fx + fpw);
		//H = std::max(H, abs(c.tau * fpw))
		// H = H; // just a reminder that simply setting H to tau has to be tested.
	}
	grad += fprime * grad_acc;

	float deltaX = xlr * grad / H;
	//float deltaX = std::clamp(xlr * grad / H, -.2f, .2f); 

	deltaX *= 1.0f - (x * deltaX > 0.f) * xReg;
	x += deltaX;
	epsilon = x - mu;

	float Hb = tau + bx_precision;	// L1, not the real H
	float deltaB = wxlr * (epsilon * tau + bx_precision * (bx_mean - bx_variate)) / Hb;

	bx_variate += deltaB;
	mu += deltaB;
	epsilon = x - mu;


	float new_fx = tanhf(x);
	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

		// between x and w update. Put like this for clarity, but could be updated only at the end of the loop iteration for efficiency.
		c.mu += (new_fx - fx) * wx_variates[k]; 
		c.epsilon = c.x - c.mu; 

		float Hw = c.tau * new_fx * new_fx + wx_precisions[k];	// L1, not the real H
		
		float deltaW = wxlr * (c.epsilon * c.tau * new_fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hw;
		deltaW *= 1.0f - (wx_variates[k] * deltaW > 0.f) * wxReg;

		c.mu += new_fx * deltaW;
		c.epsilon = c.x - c.mu;

		wx_variates[k] += deltaW;
	}
	fx = new_fx;
}


void Node::synchronousGradientStep()
{
	float grad = -epsilon * tau;
	float fprime = 1.0f - powf(fx, 2.f);
	float grad_acc = .0f;

	float H = tau;

	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];
		grad_acc += c.epsilon * c.tau * wx_variates[k];

		float fpw = wx_variates[k] * fprime;
		H += abs(c.tau * fpw * (c.epsilon * fx + fpw));  // L1 H
		//H += c.tau * fpw * (c.epsilon * fx + fpw);
		//H = std::max(H, abs(c.tau * fpw))
		// H = H; // just a reminder that simply setting H to tau has to be tested.
	}
	grad += fprime * grad_acc;

	float deltaX = xlr * grad / H;
	//float deltaX = std::clamp(xlr * grad / H, -.2f, .2f); 

	deltaX *= 1.0f - (x * deltaX > 0.f) * xReg;
	x += deltaX;

	float Hb = tau + bx_precision;	// L1, not the real H
	float deltaB = wxlr * ((epsilon+deltaX) * tau + bx_precision * (bx_mean - bx_variate)) / Hb;

	bx_variate += deltaB;


	fx = tanhf(x);
	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

		float Hw = c.tau * fx * fx + wx_precisions[k];	// L1, not the real H
		float deltaW = wxlr * (c.epsilon * c.tau * fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hw;
		deltaW *= 1.0f - (wx_variates[k] * deltaW > 0.f) * wxReg;
		wx_variates[k] += deltaW;
	}
}


void Node::calcifyWB()
{
	for (int k = 0; k < nChildren; k++)
	{
		// += observationImportance * .5f * tau[i][j] * powf(fx[i + 1][k], 2.0f); does not work, beacause already implicitly present in the value w takes.
		//  But having the importance increase of a constant quantity all the time feels wrong... TODO priority
		wx_precisions[k] += observationImportance;
		wx_precisions[k] *= certaintyDecay; // TODO heuristics here 

		wx_means[k] = wx_variates[k];
	}
	
	bx_precision += observationImportance;
	bx_precision *= certaintyDecay;

	bx_mean = bx_variate;
}


void Node::setActivation(float newX)
{
	epsilon = epsilon + newX - x;
	x = newX;
	fx = tanhf(x);
}