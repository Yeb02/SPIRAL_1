#pragma once

// one and only one must be active:
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


//#define DYNAMIC_PRECISIONS  // Do not use until the reason why on the retrocausal task mus/bs are drawn to infinity. Does not happen with FIXED_PRECISIONS_BUT_CONTRIBUTE.
#ifndef DYNAMIC_PRECISIONS
//#define FIXED_PRECISIONS_BUT_CONTRIBUTE 
#endif


#define WBX_IGNORE_TAU 


#define ASYNCHRONOUS_UPDATES 



// one and only one must be active:
#define TANH 
//#define QSIGMOIDE
//#define ID 

#ifdef TANH
#define F(x) tanhf(x)			   // F' = (1-tanhf²) = 1 - F²
#elif defined(QSIGMOIDE)
#define F(x) (.5f*tanhf(x) + .5f)  // F' = .5(1-tanhf²) = .5 - 2 * (F-.5)². Sigmoide is .5f*tanhf(.5*x) + .5f
#elif defined(ID)
#define F(x) x 
#endif




#define ANALYTICAL_X

#ifdef ANALYTICAL_X
#undef F
#define F(x) std::clamp(x, a, b) // a, b are defined at the top of node.cpp. -1 and 1 typically.
#endif


// For use in the analytical update only. Not exactly an L1 reg, because no analytical form. Just a small incentive 
// for picking x = 0. The L2 regularisation is still used.
//#define REGXL1


// TODO no analytical form, maybe like for the activations a comparison of E0 and Ew*, picking 0 if E0-bonus<Ew*.
//#define REGWL1


// Influence on results ?
// Immediatly sets x to mu whenever mu changes for the nodes that must be inferred and have no children. Typically the label.
#define FREE_NODES


// one and only one must be active:
//#define REGWX 1.0f
#define REGWX (fi * fi)
//#define REGWX (epsilon * epsilon * fi * fi)

// one and only one must be active:
//#define REGWT 1.0f
#define REGWT (fi * fi)




// Not exactly vanilla topology, as the label and datapoint are still both at the bottom ( TODO an option to test label on top as well )
// The wlr parameter is in the node::predictiveCodingWxGradientStep() function's definition.
// Make sure that dynamicTopology is set to false in main !

//#define VANILLA_PREDICTIVE_CODING

#ifdef VANILLA_PREDICTIVE_CODING // Do not modify what is in this #if. Some of the undefined directives are compatible, but this is only for benchmarks so dont bother

#define ASYNCHRONOUS_UPDATES
#define NO_SECOND_ORDER 
#undef SECOND_ORDER_L1
#undef ANALYTICAL_X
#undef DYNAMIC_PRECISIONS
#undef FIXED_PRECISIONS_BUT_CONTRIBUTE
#undef ASYNCHRONOUS_UPDATES
#undef REGWX
#define REGWX 1.0f
#undef F
#define F(x) tanhf(x)

#endif


