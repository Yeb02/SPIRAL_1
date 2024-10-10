#pragma once

// For FCNs
#define LABEL_IS_DATAPOINT


// one and only one must be active:
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


//#define DYNAMIC_PRECISIONS 
#ifdef DYNAMIC_PRECISIONS
#define WBX_IGNORE_TAU 
#endif



#define ASYNCHRONOUS_UPDATES // TODO re-test synchronous, second order should mitigate divergence issues


#define F(x) (2.0f*tanhf(x) + 1.f) // F' = 2(1-tanhf²) = 2 - .5 * (F-1)². Not quite sigmoide.

// one and only one must be active:
//#define REGWX 1.0f
#define REGWX (fi * fi)
//#define REGWX (epsilon * epsilon * fi * fi)

// one and only one must be active:
//#define REGWT 1.0f
#define REGWT (fi * fi)

// one and only one must be active:
#define REGBT .0f
//#define REGBT 1.0f


