#include "Node.h"

// Recommended values:

float Node::xlr = .8f;
float Node::wxlr = .1f;
float Node::wtlr = .1f;

float Node::wxPriorStrength = .2f;
float Node::wtPriorStrength = .2f;

float Node::observationImportance = 1.0f;
float Node::certaintyDecay = 1.0f;
float Node::certaintyLimit = 100.0f;

float Node::xReg = .01f;
float Node::wxReg = .01f;
float Node::wtReg = .01f;
float Node::btReg = .2f;



Node::Node(int _nChildren, Node** _children) :
	nChildren(_nChildren)
{
	

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
	bx_energy = 0.f;
	wx_energies = new float[nChildren];
	std::fill(wx_energies, wx_energies + nChildren, .0f);
	resetFlag = 1.f;

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

#ifdef DYNAMIC_PRECISIONS
const float eta = .0f;
#endif 

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
		grad_acc += c.tau * c.epsilon * (- wx_variates[k] + wt_variates[k] * c.epsilon) - wt_variates[k];
		
		grad_acc += eta * wt_variates[k] * c.tau;
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

void Node::WBGradientStep()
{
	float Hbx = tau + bx_precision;	
	float gradbx = bx_precision * (bx_variate - bx_mean) - epsilon * tau;
	float deltaBx = - wxlr * gradbx / Hbx;

	bx_variate += deltaBx;
	mu += deltaBx;
	epsilon = x - mu;
	

#ifdef DYNAMIC_PRECISIONS
	float te2 = .5f * epsilon * epsilon * tau;
	float Hbt = btReg * REGBT + te2 + bt_precision;
	Hbt += eta * tau;
	float gradbt = btReg * REGBT * (bt_variate-.5f)
		+ bt_precision * (bt_variate - bt_mean) + te2 - 1.f;
	gradbt += eta * tau;
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
		float Hwtk = wtReg * REGWT
			+ wx_precisions[k] + te2k * fx * fx;	
		Hwtk += eta * c.tau * fx * fx;
		float gradwtk = wtReg * wt_variates[k] * REGWT
			+ wx_precisions[k] * (wt_variates[k] - wt_means[k]) + te2k * fx * fx - fx;
		gradwtk += eta * c.tau * fx;
		float deltaWtk = - wtlr * gradwtk / Hwtk;
		wt_variates[k] += deltaWtk;
#ifdef ASYNCHRONOUS_UPDATES
		c.t += fx * deltaWtk;
		c.tau = expf(c.t);
#endif
#endif


		float Hwxk = wxReg * REGWX
			+ wx_precisions[k] + c.tau * fx * fx;
		float gradwxk = wxReg * wx_variates[k] * REGWX
			+ wx_precisions[k] * (wx_variates[k] - wx_means[k]) - c.tau * c.epsilon * fx;
		float deltaWxk = - wxlr * gradwxk / Hwxk;
		wx_variates[k] += deltaWxk;
#ifdef ASYNCHRONOUS_UPDATES
		c.mu += fx * deltaWxk;
		c.epsilon = c.x - c.mu;
#endif
	}
}



void Node::calcifyWB()
{

	for (int k = 0; k < nChildren; k++)
	{
		float deltaWxk2 = powf(wx_variates[k] - wx_means[k], 2.0f);
		float decayWxk = std::max(expf(-certaintyDecay * deltaWxk2 * wx_precisions[k]), .8f);

		wx_energies[k] = children[k]->resetFlag * wx_energies[k] * decayWxk + deltaWxk2; // without taus
		//wx_energies[k] = children[k]->resetFlag * wx_energies[k] * decayWxk + deltaWxk2 * wx_precisions[k]; // with taus

		wx_precisions[k] = wx_precisions[k] * decayWxk + observationImportance * fx * fx;
		wx_precisions[k] = std::min(wx_precisions[k], certaintyLimit);
		wx_means[k] = wx_variates[k];


#ifdef DYNAMIC_PRECISIONS
		float deltaWtk = wt_variates[k] - wt_means[k];
		float decayWtk = std::max(expf(-certaintyDecay * powf(deltaWtk, 2.0f) * wt_precisions[k]), .8f);

		wt_precisions[k] = wt_precisions[k] * decayWtk + observationImportance * fx * fx;
		wt_precisions[k] = std::min(wt_precisions[k], certaintyLimit);
		wt_means[k] = wt_variates[k];
#endif
	}
	
	float deltaBx2 = powf(bx_variate - bx_mean, 2.0f);
	float decayBx = std::max(expf(-certaintyDecay * deltaBx2 * bx_precision), .8f);

	bx_energy = resetFlag * bx_energy * decayBx + deltaBx2 * (1.f + powf(bx_precision / tau, 2.f)); // without taus
	//bx_energy = resetFlag * bx_energy * decayBx + deltaBx2 * bx_precision * (1.f + bx_precision / tau); // with taus
	

	bx_precision = bx_precision * decayBx + observationImportance;
	bx_precision = std::min(bx_precision, certaintyLimit);
	bx_mean = bx_variate;

	

#ifdef DYNAMIC_PRECISIONS
	float deltaBt = bt_variate - bt_mean;
	float decayBt = std::max(expf(-certaintyDecay * powf(deltaBt, 2.0f) * bt_precision), .8f);

	bt_precision = bt_precision * decayBt + observationImportance;
	bt_precision = std::min(bt_precision, certaintyLimit);
	bt_mean = bt_variate;
#endif
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


void Node::prepareToReceiveEnergies()
{
	accumulatedEnergy = bx_energy;
}

void Node::transmitEnergies()
{
	for (int k = 0; k < nChildren; k++)
	{
		children[k]->accumulatedEnergy += wx_energies[k];
	}
}
