#include "Node.h"

// Recommended values:

float Node::xlr = 1.f;
float Node::wxlr = .1f;
float Node::wtlr = .1f;

float Node::wxPriorStrength = .2f;
float Node::wtPriorStrength = .2f;
float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .95f;

float Node::xReg = .01f;
float Node::wxReg = .01f;
float Node::wtReg = .01f;


Node::Node(int _nChildren, Node** _children) :
	nChildren(_nChildren)
{
	

	children = new Node*[nChildren];
	std::copy(_children, _children + nChildren, children);

	bx_mean = NORMAL_01 * .05f;
	bx_precision = wxPriorStrength;
	bx_variate = bx_mean;

	wx_means = new float[nChildren];
	wx_precisions = new float[nChildren];
	wx_variates = new float[nChildren];

	std::fill(wx_precisions, wx_precisions + nChildren, wxPriorStrength);
	for (int i = 0; i < nChildren; i++) wx_means[i] = NORMAL_01 * .05f;
	std::copy(wx_means, wx_means + nChildren, wx_variates);

	x = NORMAL_01 * .05f;
	fx = tanhf(x);

#ifdef DYNAMIC_PRECISIONS

	bt_mean = NORMAL_01 * .001f;
	bt_precision = wtPriorStrength;
	bt_variate = bt_mean;

	wt_means = new float[nChildren];
	wt_precisions = new float[nChildren];
	wt_variates = new float[nChildren];

	std::fill(wt_precisions, wt_precisions + nChildren, wtPriorStrength);
	for (int i = 0; i < nChildren; i++) wt_means[i] = NORMAL_01 * .001f;
	std::copy(wt_means, wt_means + nChildren, wt_variates);

#else
	tau = 1.0f;
#endif

	prepareToReceivePredictions();
	transmitPredictions();
	computeLocalQuantities();
}

Node::~Node()
{
	delete[] children;
	delete[] wx_variates;
	delete[] wx_means;
	delete[] wx_precisions;


#ifdef DYNAMIC_PRECISIONS
	delete[] wt_variates;
	delete[] wt_means;
	delete[] wt_precisions;
#endif
}


// Dont forget to change the other 2 functions below when touching this one !
void Node::asynchronousGradientStep()
{
	//Everything up to date initially
	

	
	{
		float grad = -epsilon * tau;
		float fprime = 1.0f - powf(fx, 2.f);
		float grad_acc = .0f;

		float H = tau;

		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];
			grad_acc += c.epsilon * c.tau * wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
			grad_acc += wt_variates[k] * (c.e - 1.f);
#endif


			float fpw = wx_variates[k] * fprime;

#ifdef NO_SECOND_ORDER
			//
#elif defined SECOND_ORDER_TAU
			//
#elif defined SECOND_ORDER_MAX
			H = std::max(H, abs(c.tau * fpw));
#elif defined SECOND_ORDER_L1
			H += abs(c.tau * fpw * (c.epsilon * fx + fpw));
#elif defined SECOND_ORDER_EXACT
			H += c.tau * fpw * (c.epsilon * fx + fpw);
#endif
		}
		grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
		H = 1.f;
#endif

		float deltaX = xlr * grad / H;
#ifdef SECOND_ORDER_EXACT
		deltaX = std::clamp(xlr * grad / H, -1.f, 1.f);
#endif


		deltaX *= 1.0f - ((x * deltaX) > 0.f) * xReg;
		x += deltaX;

	}
	// X changed
	// epsilon, fx, e not up to date


	{
		epsilon = x - mu;

		float Hbx = tau + bx_precision;	// L1, not the real H
		float deltaBX = wxlr * (epsilon * tau + bx_precision * (bx_mean - bx_variate)) / Hbx;
		bx_variate += deltaBX;
		mu += deltaBX;
		epsilon = x - mu;
	}
	// BX changed
	// fx, e not up to date


#ifdef DYNAMIC_PRECISIONS
	{
		e = .5f * epsilon * epsilon * tau;

		float Hbt = e + bt_precision;	// L1, not the real H
		float deltaBT = wtlr * (1.f - e + bx_precision * (bt_mean - bt_variate)) / Hbt;

		bt_variate += deltaBT;
		t += deltaBT;
		tau = expf(t);
	}
	// BT changed
	// fx, e not up to date
#endif



	{
#ifdef DYNAMIC_PRECISIONS
		e = .5f * epsilon * epsilon * tau;
#endif

		float deltaFX = tanhf(x) - fx;
		fx += deltaFX;
		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];

			
			c.mu += deltaFX * wx_variates[k];
			c.epsilon = c.x - c.mu;


#ifdef DYNAMIC_PRECISIONS
			c.t += deltaFX * wt_variates[k];
			c.tau = expf(c.t);
			c.e = .5f * c.epsilon * c.epsilon * c.tau;

			float Hwt = c.e * fx * fx + wx_precisions[k];	// L1 is also the real H (E is convex in wt)
			float deltaWt = wtlr * ((1.f - c.e) * fx + wt_precisions[k] * (wt_means[k] - wt_variates[k])) / Hwt; 
			deltaWt *= 1.0f - ((wt_variates[k] * deltaWt) > 0.f) * wtReg;

			wt_variates[k] += deltaWt;
			c.t += fx * deltaWt;
			c.tau = expf(c.t);
#endif



			float Hwx = c.tau * fx * fx + wx_precisions[k];	// L1, not the real H
			float deltaWx = wxlr * (c.epsilon * c.tau * fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hwx;
			deltaWx *= 1.0f - ((wx_variates[k] * deltaWx) > 0.f) * wxReg;

			wx_variates[k] += deltaWx;
			c.mu += fx * deltaWx;
			c.epsilon = c.x - c.mu;

#ifdef DYNAMIC_PRECISIONS
			c.e = .5f * c.epsilon * c.epsilon * c.tau;
#endif
		}
	}
	// All local quantities up to date
	// Children mu and t updated
	// All children's quantities up to date
	
}

void Node::asynchronousGradientStep_X_only()
{
	//Everything up to date initially



	{
		float grad = -epsilon * tau;
		float fprime = 1.0f - powf(fx, 2.f);
		float grad_acc = .0f;

		float H = tau;

		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];
			grad_acc += c.epsilon * c.tau * wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
			grad_acc += wt_variates[k] * (c.e - 1.f);
#endif


			float fpw = wx_variates[k] * fprime;

#ifdef NO_SECOND_ORDER
			//
#elif defined SECOND_ORDER_TAU
			//
#elif defined SECOND_ORDER_MAX
			H = std::max(H, abs(c.tau * fpw));
#elif defined SECOND_ORDER_L1
			H += abs(c.tau * fpw * (c.epsilon * fx + fpw));
#elif defined SECOND_ORDER_EXACT
			H += c.tau * fpw * (c.epsilon * fx + fpw);
#endif
		}
		grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
		H = 1.f;
#endif

		float deltaX = xlr * grad / H;
#ifdef SECOND_ORDER_EXACT
		deltaX = std::clamp(xlr * grad / H, -1.f, 1.f);
#endif


		deltaX *= 1.0f - ((x * deltaX) > 0.f) * xReg;
		x += deltaX;
		epsilon = x - mu;
#ifdef DYNAMIC_PRECISIONS
		e = .5f * epsilon * epsilon * tau;
#endif
	}
	// X changed
	// fx not up to date


	{

		float deltaFX = tanhf(x) - fx;
		fx += deltaFX;
		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];

			c.mu += deltaFX * wx_variates[k];
			c.epsilon = c.x - c.mu;

#ifdef DYNAMIC_PRECISIONS
			c.t += deltaFX * wt_variates[k];
			c.tau = expf(c.t);
			c.e = .5f * c.epsilon * c.epsilon * c.tau;
#endif
		}
	}
	// All local quantities up to date
	// Children mu and t updated
	// All children's quantities up to date

}

void Node::asynchronousGradientStep_WB_only()
{
	//Everything up to date initially


	{
		float Hbx = tau + bx_precision;	// L1, not the real H
		float deltaBX = wxlr * (epsilon * tau + bx_precision * (bx_mean - bx_variate)) / Hbx;
		bx_variate += deltaBX;
		mu += deltaBX;
		epsilon = x - mu;
	}
	// BX changed
	// e not up to date


#ifdef DYNAMIC_PRECISIONS
	{
		e = .5f * epsilon * epsilon * tau;

		float Hbt = e + bt_precision;	// L1, not the real H
		float deltaBT = wtlr * (1.f - e + bx_precision * (bt_mean - bt_variate)) / Hbt;

		bt_variate += deltaBT;
		t += deltaBT;
		tau = expf(t);
		e = .5f * epsilon * epsilon * tau;
	}
	// BT changed
	// Everything up to date
#endif



	{
		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];


#ifdef DYNAMIC_PRECISIONS
			float Hwt = c.e * fx * fx + wx_precisions[k];	// L1 is also the real H (E is convex in wt)
			float deltaWt = wtlr * ((1.f - c.e) * fx + wt_precisions[k] * (wt_means[k] - wt_variates[k])) / Hwt;
			deltaWt *= 1.0f - ((wt_variates[k] * deltaWt) > 0.f) * wtReg;

			wt_variates[k] += deltaWt;
			c.t += fx * deltaWt;
			c.tau = expf(c.t);
#endif



			float Hwx = c.tau * fx * fx + wx_precisions[k];	// L1, not the real H
			float deltaWx = wxlr * (c.epsilon * c.tau * fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hwx;
			deltaWx *= 1.0f - ((wx_variates[k] * deltaWx) > 0.f) * wxReg;

			wx_variates[k] += deltaWx;
			c.mu += fx * deltaWx;
			c.epsilon = c.x - c.mu;

#ifdef DYNAMIC_PRECISIONS
			c.e = .5f * c.epsilon * c.epsilon * c.tau;
#endif
		}
	}
	// All local quantities up to date
	// Children mu and t updated
	// All children's quantities up to date

}


// Dont forget to change the other 2 functions below when touching this one !
void Node::synchronousGradientStep()
{
	//Everything up to date initially



	{
		float grad = -epsilon * tau;
		float fprime = 1.0f - powf(fx, 2.f);
		float grad_acc = .0f;

		float H = tau;

		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];
			grad_acc += c.epsilon * c.tau * wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
			grad_acc += wt_variates[k] * (c.e - 1.f);
#endif


			float fpw = wx_variates[k] * fprime;

#ifdef NO_SECOND_ORDER
			//
#elif defined SECOND_ORDER_TAU
			//
#elif defined SECOND_ORDER_MAX
			H = std::max(H, abs(c.tau * fpw));
#elif defined SECOND_ORDER_L1
			H += abs(c.tau * fpw * (c.epsilon * fx + fpw));
#elif defined SECOND_ORDER_EXACT
			H += c.tau * fpw * (c.epsilon * fx + fpw);
#endif
		}
		grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
		H = 1.f;
#endif

		float deltaX = xlr * grad / H;
#ifdef SECOND_ORDER_EXACT
		deltaX = std::clamp(xlr * grad / H, -1.f, 1.f);
#endif


		deltaX *= 1.0f - ((x * deltaX) > 0.f) * xReg;
		x += deltaX;

	}
	// X changed
	// epsilon, fx, e not up to date

	{
		epsilon = x - mu;
		float Hbx = tau + bx_precision;	// L1, not the real H
		float deltaBX = wxlr * (epsilon * tau + bx_precision * (bx_mean - bx_variate)) / Hbx;
		bx_variate += deltaBX;
		mu += deltaBX;
		epsilon = x - mu;
	}
	// BX changed
	// fx, e not up to date


#ifdef DYNAMIC_PRECISIONS
	{
		e = .5f * epsilon * epsilon * tau;

		float Hbt = e + bt_precision;	// L1, not the real H
		float deltaBT = wtlr * (1.f - e + bx_precision * (bt_mean - bt_variate)) / Hbt;

		bt_variate += deltaBT;
		t += deltaBT;
		tau = expf(t);
		e = .5f * epsilon * epsilon * tau;
	}
	// BT changed
	// fx not up to date
#endif



	{
		float deltaFX = tanhf(x) - fx;
		fx += deltaFX;
		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];


#ifdef DYNAMIC_PRECISIONS
			float Hwt = c.e * fx * fx + wx_precisions[k];	// L1 is also the real H (E is convex in wt)
			float deltaWt = wtlr * ((1.f - c.e) * fx + wt_precisions[k] * (wt_means[k] - wt_variates[k])) / Hwt;
			deltaWt *= 1.0f - ((wt_variates[k] * deltaWt) > 0.f) * wtReg;
			wt_variates[k] += deltaWt;
#endif



			float Hwx = c.tau * fx * fx + wx_precisions[k];	// L1, not the real H
			float deltaWx = wxlr * (c.epsilon * c.tau * fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hwx;
			deltaWx *= 1.0f - ((wx_variates[k] * deltaWx) > 0.f) * wxReg;
			wx_variates[k] += deltaWx;
		}
	}
	// All local quantities up to date but mu and t do not correspond to the parent's activations.
}

void Node::synchronousGradientStep_X_only()
{
	//Everything up to date initially

	{
		float grad = -epsilon * tau;
		float fprime = 1.0f - powf(fx, 2.f);
		float grad_acc = .0f;

		float H = tau;

		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];
			grad_acc += c.epsilon * c.tau * wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
			grad_acc += wt_variates[k] * (c.e - 1.f);
#endif


			float fpw = wx_variates[k] * fprime;

#ifdef NO_SECOND_ORDER
			//
#elif defined SECOND_ORDER_TAU
			//
#elif defined SECOND_ORDER_MAX
			H = std::max(H, abs(c.tau * fpw));
#elif defined SECOND_ORDER_L1
			H += abs(c.tau * fpw * (c.epsilon * fx + fpw));
#elif defined SECOND_ORDER_EXACT
			H += c.tau * fpw * (c.epsilon * fx + fpw);
#endif
		}
		grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
		H = 1.f;
#endif

		float deltaX = xlr * grad / H;
#ifdef SECOND_ORDER_EXACT
		deltaX = std::clamp(xlr * grad / H, -1.f, 1.f);
#endif


		deltaX *= 1.0f - ((x * deltaX) > 0.f) * xReg;
		x += deltaX;
		epsilon = x - mu;
#ifdef DYNAMIC_PRECISIONS
		e = .5f * epsilon * epsilon * tau;
#endif

		fx = tanhf(x);
	}
	// X changed
	// All local quantities up to date but mu and t do not correspond to the parent's activations.
}

void Node::synchronousGradientStep_WB_only()
{
	//Everything up to date initially


	{
		float Hbx = tau + bx_precision;	// L1, not the real H
		float deltaBX = wxlr * (epsilon * tau + bx_precision * (bx_mean - bx_variate)) / Hbx;
		bx_variate += deltaBX;
		mu += deltaBX;
		epsilon = x - mu;
	}
	// BX changed
	// e not up to date


#ifdef DYNAMIC_PRECISIONS
	{
		e = .5f * epsilon * epsilon * tau;

		float Hbt = e + bt_precision;	// L1, not the real H
		float deltaBT = wtlr * (1.f - e + bx_precision * (bt_mean - bt_variate)) / Hbt;

		bt_variate += deltaBT;
		t += deltaBT;
		tau = expf(t);
		e = .5f * epsilon * epsilon * tau;
	}
	// BT changed
	// everything up to date
#endif



	{
		for (int k = 0; k < nChildren; k++)
		{
			Node& c = *children[k];


#ifdef DYNAMIC_PRECISIONS
			float Hwt = c.e * fx * fx + wx_precisions[k];	// L1 is also the real H (E is convex in wt)
			float deltaWt = wtlr * ((1.f - c.e) * fx + wt_precisions[k] * (wt_means[k] - wt_variates[k])) / Hwt;
			deltaWt *= 1.0f - ((wt_variates[k] * deltaWt) > 0.f) * wtReg;
			wt_variates[k] += deltaWt;
#endif

			float Hwx = c.tau * fx * fx + wx_precisions[k];	// L1, not the real H
			float deltaWx = wxlr * (c.epsilon * c.tau * fx + wx_precisions[k] * (wx_means[k] - wx_variates[k])) / Hwx;
			deltaWx *= 1.0f - ((wx_variates[k] * deltaWx) > 0.f) * wxReg;
			wx_variates[k] += deltaWx;
		}
	}
	// All local quantities up to date but mu and t do not correspond to the parent's activations.
}


void Node::calcifyWB()
{
	for (int k = 0; k < nChildren; k++)
	{
		// += observationImportance * .5f * tau[i][j] * powf(fx[i + 1][k], 2.0f); does not work, beacause already implicitly present in the value w takes.
		//  But having the importance increase of a constant quantity all the time feels wrong... TODO priority
		wx_precisions[k] = (wx_precisions[k] + observationImportance) * certaintyDecay;

		wx_means[k] = wx_variates[k];
	}
	
	bx_precision += observationImportance;
	bx_precision *= certaintyDecay;

	bx_mean = bx_variate;
}



void Node::setActivation(float newX)
{
	x = newX;
	epsilon = x - mu;

#ifdef DYNAMIC_PRECISIONS
	e = .5f * epsilon * epsilon * tau;
#endif

	float deltaFX = tanhf(x) - fx;
	fx += deltaFX;

	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

		c.mu += deltaFX * wx_variates[k];

#ifdef DYNAMIC_PRECISIONS
		c.t += deltaFX * wt_variates[k];
#endif
	}
}

void Node::prepareToReceivePredictions()
{
	mu = bx_variate;
#ifdef DYNAMIC_PRECISIONS
	t = bt_variate;
#endif
}

void Node::transmitPredictions()
{
	for (int k = 0; k < nChildren; k++)
	{
		children[k]->mu += fx * wx_variates[k];
#ifdef DYNAMIC_PRECISIONS
		children[k]->t += fx * wt_variates[k];
#endif
	}
}

void Node::computeLocalQuantities() 
{
	epsilon = x - mu;

#ifdef DYNAMIC_PRECISIONS
	tau = expf(t);
	e = .5f * epsilon * epsilon * tau;
#endif
}