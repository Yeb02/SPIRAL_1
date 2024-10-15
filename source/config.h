#pragma once

// For FCNs
#define LABEL_IS_DATAPOINT


// one and only one must be active:
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


#define DYNAMIC_PRECISIONS 
#ifndef DYNAMIC_PRECISIONS
#define FIXED_PRECISIONS_BUT_CONTRIBUTE
#endif


#define WBX_IGNORE_TAU 


#define ASYNCHRONOUS_UPDATES 



#define TANH // works significantly better than pseudoSigmoide.
#ifdef TANH
#define F(x) tanhf(x)			   // F' = (1-tanhf²) = 1 - F²
#else // quasiSigmoide
#define F(x) (.5f*tanhf(x) + .5f)  // F' = .5(1-tanhf²) = .5 - 2 * (F-.5)². Sigmoide is .5f*tanhf(.5*x) + .5f
#endif


// one and only one must be active:
//#define REGWX 1.0f
#define REGWX (fi * fi)
//#define REGWX (epsilon * epsilon * fi * fi)

// one and only one must be active:
//#define REGWT 1.0f
#define REGWT (fi * fi)


// Not exactly vanilla topology, as the label and datapoint are still both at the bottom ( TODO an option to test label on top as well )
// Make sure that dynamicTopology is set to false in main !
#define VANILLA_PREDICTIVE_CODING
#ifdef VANILLA_PREDICTIVE_CODING // Do not modify what is in this #if.

#define ASYNCHRONOUS_UPDATES
#define NO_SECOND_ORDER 
#undef SECOND_ORDER_L1
#undef DYNAMIC_PRECISIONS
#undef FIXED_PRECISIONS_BUT_CONTRIBUTE
#undef ASYNCHRONOUS_UPDATES
#undef TANH // This one can be changed
#undef REGWX
#define REGWX 1.0f

#endif


