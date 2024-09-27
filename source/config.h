#pragma once

// For FCNs
#define LABEL_IS_DATAPOINT


// one and only one must be active:
//#define NO_SECOND_ORDER
//#define SECOND_ORDER_TAU
//#define SECOND_ORDER_MAX
#define SECOND_ORDER_L1


#define DYNAMIC_PRECISIONS 


#define ASYNCHRONOUS_UPDATES // TODO test synchronous, second order should mitigate divergence issues
#ifdef ASYNCHRONOUS_UPDATES
#define RANDOM_UPDATE_ORDER
#endif


#define F(x) (2.0f*tanhf(x) + 1.f) // F' = 2(1-tanhf²) = 2 - .5 * (F-1)²

// one and only one must be active:
//#define REGX 1.0f
//#define REGX (fx * fx)
#define REGWX (c.epsilon * c.epsilon * fx * fx)

// one and only one must be active:
//#define REGWT 1.0f
#define REGWT fx * fx

// one and only one must be active:
#define REGBT 1.0f


