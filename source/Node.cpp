#include "Node.h"

// Recommended values:

float Node::xlr = .8f;
float Node::wxlr = .1f;
float Node::wtlr = .1f;

float Node::lambda1 = 1.f;
float Node::lambda2 = 1.f;

float Node::wxPriorStrength = .2f;
float Node::wtPriorStrength = .2f;

float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .01f;

float Node::xReg = .01f;
float Node::wxReg = .01f;
float Node::wtReg = .01f;
float Node::btReg = .2f;



Node::Node(int _nChildren, Node** _children) :
	nChildren(_nChildren)
{
	
	parents.resize(0); // for completeness
	inParentsListIDs.resize(0); // for completeness

	children = new Node*[nChildren];
	std::copy(_children, _children + nChildren, children);

	bx_mean = NORMAL_01 * .2f;
	bx_precision = wxPriorStrength;
	bx_variate = bx_mean;

	wx_means = new float[nChildren];
	wx_precisions = new float[nChildren];
	wx_variates = new float[nChildren];

	std::fill(wx_precisions, wx_precisions + nChildren, wxPriorStrength);
	for (int i = 0; i < nChildren; i++) wx_means[i] = NORMAL_01 * .2f;
	std::copy(wx_means, wx_means + nChildren, wx_variates);

	mu = bx_mean;
	x = NORMAL_01 * .2f;
	fx = F(x);


	accumulatedEnergy = .0f;


#ifdef DYNAMIC_PRECISIONS

	bt_mean = .5f + NORMAL_01 * .1f; // -1 because typical values are in [-1, 1], so N(0, tau=exp(.5)) is better than N(0, 1)
	bt_precision = wtPriorStrength;
	bt_variate = bt_mean;

	wt_means = new float[nChildren];
	wt_precisions = new float[nChildren];
	wt_variates = new float[nChildren];

	std::fill(wt_precisions, wt_precisions + nChildren, wtPriorStrength);
	for (int i = 0; i < nChildren; i++) wt_means[i] = NORMAL_01 * .025f;
	std::copy(wt_means, wt_means + nChildren, wt_variates);
	t = bt_mean;

#else
	tau = 1.0f;
#endif

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


void Node::XGradientStep()
{
	float grad = epsilon * tau + x * xReg;
	float fprime = 2.f - .5f * powf(fx - 1.f, 2.f);
	float grad_acc = .0f;

	float H = tau + xReg;

	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

#ifdef DYNAMIC_PRECISIONS
		grad_acc += c.tau * c.epsilon * (-wx_variates[k] + wt_variates[k] * c.epsilon); // -wt_variates[k];
#else
		grad_acc -= c.tau * c.epsilon * wx_variates[k];
#endif


		float fpw = wx_variates[k] * fprime;
		float hk = c.tau * fpw * (fpw + c.epsilon * (fx-1.f));
#ifdef NO_SECOND_ORDER
		// no operation needed
#elif defined SECOND_ORDER_TAU
		// no operation needed
#elif defined SECOND_ORDER_MAX
		H = std::max(H, abs(hk));
#elif defined SECOND_ORDER_L1
		H += abs(hk);
#endif
	}
	grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
	H = 1.f;
#endif

	x -= xlr * grad / H;
	epsilon = x - mu;

	float deltaFX = F(x) - fx;
	fx += deltaFX;

#ifdef ASYNCHRONOUS_UPDATES
	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

		c.mu += deltaFX * wx_variates[k];
		c.epsilon = c.x - c.mu;

#ifdef DYNAMIC_PRECISIONS
		c.t += deltaFX * wt_variates[k];
		c.tau = expf(c.t);
#endif
	}
#endif 
}


// deprecated
void Node::WBGradientStep()
{
#ifdef WBX_IGNORE_TAU
	float Hbx = bx_precision + 1.f;
	float gradbx = bx_precision * (bx_variate - bx_mean) - epsilon;
#else
	float Hbx = bx_precision + tau;
	float gradbx = bx_precision * (bx_variate - bx_mean) - epsilon * tau;
#endif
	float deltaBx = - wxlr * gradbx / Hbx;

	bx_variate += deltaBx;
	mu += deltaBx;
	epsilon = x - mu;
	

#ifdef DYNAMIC_PRECISIONS


	float te2 = .5f * epsilon * epsilon * tau;
	float Hbt = btReg * REGBT + te2 + bt_precision;
	float gradbt = btReg * REGBT * (bt_variate-.5f)
		+ bt_precision * (bt_variate - bt_mean) + te2 - 1.f;
	float deltaBT = -wtlr * gradbt / Hbt;

	bt_variate += deltaBT;
	t += deltaBT;
	tau = expf(t);
#endif
	
	for (int k = 0; k < nChildren; k++)
	{
		Node& c = *children[k];

#ifdef DYNAMIC_PRECISIONS
		float te2k = .5f * c.epsilon * c.epsilon * c.tau;
		float Hwtk = //wtReg * REGWT +
			 wx_precisions[k] + te2k * fx * fx;	
		float gradwtk = //wtReg * wt_variates[k] * REGWT + 
			wx_precisions[k] * (wt_variates[k] - wt_means[k]) + te2k * fx * fx - fx;
		float deltaWtk = - wtlr * gradwtk / Hwtk;
		wt_variates[k] += deltaWtk;
#ifdef ASYNCHRONOUS_UPDATES
		c.t += fx * deltaWtk;
		c.tau = expf(c.t);
#endif
#endif

#ifdef WBX_IGNORE_TAU
		float Hwxk = //wxReg * REGWX+ 
			wx_precisions[k] + fx * fx;
		float gradwxk =// wxReg * wx_variates[k] * REGWX+ 
			wx_precisions[k] * (wx_variates[k] - wx_means[k]) - c.epsilon * fx;
#else
		float Hwxk = //wxReg * REGWX+ 
			wx_precisions[k] + c.tau * fx * fx;
		float gradwxk = //wxReg * wx_variates[k] * REGWX+ 
			wx_precisions[k] * (wx_variates[k] - wx_means[k]) - c.tau * c.epsilon * fx;
#endif
		float deltaWxk = - wxlr * gradwxk / Hwxk;
		wx_variates[k] += deltaWxk;
#ifdef ASYNCHRONOUS_UPDATES
		c.mu += fx * deltaWxk;
		c.epsilon = c.x - c.mu;
#endif
	}
}


void Node::setAnalyticalWX()
{
	float s1 = 1.0f / bx_precision;
	float s2 = bx_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float t1 = fi / (parents[i]->wx_precisions[id] + wxReg * REGWX);

		s1 += fi * t1;
		s2 += fi * parents[i]->wx_means[id] * parents[i]->wx_precisions[id];
	}


	epsilon = (x - s2) / (1.f + s1);
	mu = x - epsilon;


	bx_variate = epsilon/ bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wx_precisions[id];

		parents[i]->wx_variates[id] = (epsilon * fi + tau_i * parents[i]->wx_means[id])/(tau_i + wxReg * REGWX);
	}
}


void Node::setAnalyticalWT()
{
#ifdef DYNAMIC_PRECISIONS
	float s1 = 1.0f / bt_precision;
	float s2 = bt_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float t1 = fi / (parents[i]->wt_precisions[id] + wtReg * REGWT);

		s1 += fi * t1;
		s2 += fi * parents[i]->wt_means[id] * parents[i]->wt_precisions[id];
	}

	float r = .0f;
	if (nChildren != 0) {
		for (int i = 0; i < nChildren; i++)
		{
			r += children[i]->epsilon * children[i]->epsilon * powf((float)children[i]->parents.size(), -1.f);
		}
		r = logf(epsilon * epsilon / r);
	}
	float m = -2.f * logf(epsilon);
	float a = lambda1 * r + lambda2 * m;
	float b = lambda1 + lambda2;


	t = (s2 + a * s1) / (1.f + b * s1);
	tau = expf(t);

	float ambt = a - b * t;
	bx_variate = ambt / bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wt_precisions[id];

		parents[i]->wt_variates[id] = (ambt*fi + tau_i * parents[i]->wt_means[id]) / (tau_i + wtReg * REGWT);
	}
#endif
}


void Node::calcifyWB()
{

	for (int k = 0; k < nChildren; k++)
	{
		wx_precisions[k] = wx_precisions[k] * (1.0f - fx*fx*certaintyDecay) + observationImportance * fx * fx;
		wx_means[k] = wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
		wt_precisions[k] = wt_precisions[k] * (1.0f - fx * fx * certaintyDecay) + observationImportance * fx * fx;
		wt_means[k] = wt_variates[k];
#endif
	}
	
	bx_precision = bx_precision * (1.0f - certaintyDecay) + observationImportance;
	bx_mean = bx_variate;

	

#ifdef DYNAMIC_PRECISIONS
	bt_precision = bt_precision * (1.0f - certaintyDecay) + observationImportance * (lambda1 + lambda2);
	bt_mean = bt_variate;
#endif


	accumulatedEnergy = accumulatedEnergy * .99f + epsilon * epsilon;
}



void Node::setActivation(float newX)
{
	x = newX;
	epsilon = x - mu;

	float deltaFX = F(x) - fx;
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
#endif
}

