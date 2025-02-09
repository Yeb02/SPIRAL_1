#pragma once

// one and only one must be active. Used when X updates are gradient based.
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


// If enabled, inference is a coordinate ascent performed on the activation, with a random order each sweep.
// If disabled, the X updates are "parallel", i.e. virtually simultaneous across the network, like in vanilla predictive coding.
//#define ASYNCHRONOUS_UPDATES 



// one and only one must be active:
//#define TANH 
//#define QSIGMOIDE   // performs significantly worse than the other and the analytical update.
#define ID 

#ifdef TANH
#define F(x) tanhf(x)			   // F' = (1-tanhf�) = 1 - F�
#elif defined(QSIGMOIDE)
#define F(x) (.5f*tanhf(x) + .5f)  // F' = .5(1-tanhf�) = .5 - 2 * (F-.5)�. Sigmoide is .5f*tanhf(.5*x) + .5f
#elif defined(ID)
#define F(x) (x)
#endif




#define ANALYTICAL_X

#ifdef ANALYTICAL_X
#undef F
#define F(x) std::clamp((x), a, b) // a, b are defined at the top of node.cpp. -1 and 1 typically.
#endif



// one and only one must be active:
//#define REGWX 1.0f
#define REGWX (fi * fi) // best results, and most sensical
//#define REGWX (epsilon * epsilon * fi * fi)



// Not compatible with anything. Tried to find an alternative, less chaotic descent algorithm that would not oscillate as much.
// Results: with the "simple" implementations, fast and stable convergence to a local optimum. Better results with the "complex"
// implementation, almost on par with the standard descent, but reintroduces oscillations... So pointless ?
#if defined(ANALYTICAL_X) && !defined(ASYNCHRONOUS_UPDATES)
//#define INDIRECT_DESCENT
#endif

// Influence on results ? Seems like none ... Somewhat surprising, insight may be gained from understanding why.
// Immediatly sets x to mu whenever mu changes for all nodes that are not clamped and have no children. Typically
// the label at test time (or the action in future versions that will implement TD learning for RL ?).
//#define FREE_NODES    TODO NOT FINISHED, FIND FREE GROUPS IN THE GRAPH


// An attempt at mitigating H2, aka the free energy gained by lowering epsilons and increasing weights.
// Optional. Negligible computational overhead. (and gains ?)
// hyperparameters in  Group::updateTau() and  Group::Group()
//#define HOMOEPS 


// A theoretically more accurate importance term is added to the precision at calcification.
// Optional. Negligible computational overhead (and gains ?). No hyperparameter.
//#define ADVANCED_W_IMPORTANCE 


// Incompatible with ADVANCED_W_IMPORTANCE because the projected x update is complex and I am lazy. Could still work
// and should be tried TODO . Requires ANALYTICAL_X, again because I am too lazy to implement the gradient.
// Optional. Significant computational overhead. Hyperparameters in  Node::analyticalXUpdate(), gamma explicit but alpha implicit (-1 or 1)
#if defined(ANALYTICAL_X)
//////#if !defined(ADVANCED_W_IMPORTANCE) && defined(ANALYTICAL_X)
//#define LEAST_ACTION
#endif


// if defined, the regularisation of X is handled by the analytical W update only. (i.e. mu is regularized)
// Does slightly, but decisevily better than without reg, but worse than regularizing X directly ?
// Typical xReg: .2f  . Disables the direct regularization of the Xs.
//#define XREG_IN_W



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

#undef ANALYTICAL_X
#undef QSIGMOIDE
#undef ID
#define TANH
#undef F
#define F(x) tanhf(x)

#undef FREE_NODES

#undef HOMOEPS

#undef ADVANCED_W_IMPORTANCE

#undef LEAST_ACTION

#endif


