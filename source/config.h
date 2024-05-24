#pragma once



#define LABEL_IS_DATAPOINT

//#define DYNAMIC_PRECISIONS

//#define ORDINARY_GD
#ifndef ORDINARY_GD
#define PROSPECTIVE_GRAD
#ifndef PROSPECTIVE_GRAD
#define BARYGRAD
#endif
#endif

