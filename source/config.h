#pragma once

// one and only one must be active. Used when X updates are gradient based.
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


// If enabled, inference is a coordinate ascent is performed on the activation, with a random order each sweep.
// If disabled, the X updates are "parallel", i.e. virtually simultaneous across the network, like in vanilla predictive coding.
#define ASYNCHRONOUS_UPDATES 



// one and only one must be active:
//#define TANH 
//#define QSIGMOIDE   // performs significantly worse than the other and the analytical update.
#define ID 

#ifdef TANH
#define F(x) tanhf(x)			   // F' = (1-tanhf�) = 1 - F�
#elif defined(QSIGMOIDE)
#define F(x) (.5f*tanhf(x) + .5f)  // F' = .5(1-tanhf�) = .5 - 2 * (F-.5)�. Sigmoide is .5f*tanhf(.5*x) + .5f
#elif defined(ID)
#define F(x) x 
#endif




#define ANALYTICAL_X

#ifdef ANALYTICAL_X
#undef F
#define F(x) std::clamp(x, a, b) // a, b are defined at the top of node.cpp. -1 and 1 typically.
#endif



// For use in the analytical update only (TODO).  Not exactly an L1 regularization, because there is no analytical 
// closed form. The L2 regularisation is still used in parallel. Change the parameter in Node::analyticalXUpdate .
//#define REGXL1

// Not exactly an L1 reg, because there is no analytical closed form. The L2 regularisation is still used in parallel.
// Tweak the parameters in Node::setAnalyticalWX .
//#define REGWL1


// Influence on results ? Seems like none ... Somewhat surprising, insight may be gained from understanding why.
// Immediatly sets x to mu whenever mu changes for all nodes that are not clamped and have no children. Typically
// the label at test time (or the action in future versions that will implement TD learning for RL ?).
//#define FREE_NODES


// one and only one must be active:
//#define REGWX 1.0f
#define REGWX (fi * fi) // best results, and most sensical
//#define REGWX (epsilon * epsilon * fi * fi)



// An attempt at mitigating H2, aka the free energy gained by lowering epsilons and increasing weights.
// Optional, and mutually exclusive.
#define HOMOEPS_STATIC  // Relatively inexpensive, much better on paper
#ifndef HOMOEPS_STATIC
#define HOMOEPS_DYNAMIC
#endif



// The wlr parameter is in the node::predictiveCodingWxGradientStep() function's definition.
// Make sure that the right topology is picked in main. (I.e. label on top, datapoint at the bottom, topo = 3)

//#define VANILLA_PREDICTIVE_CODING

#ifdef VANILLA_PREDICTIVE_CODING // Do not modify what is in this #if. 

#undef ASYNCHRONOUS_UPDATES

#define NO_SECOND_ORDER 
#undef SECOND_ORDER_TAU
#undef SECOND_ORDER_MAX
#undef SECOND_ORDER_L1

#undef REGXL1
#undef REGWL1
#undef REGWX
#define REGWX 1.0f

#undef ANALYTICAL_X
#undef QSIGMOIDE
#undef ID
#define TANH
#undef F
#define F(x) tanhf(x)

#undef FREE_NODES

#undef HOMOEPS_STATIC
#undef HOMOEPS_DYNAMIC

#endif


