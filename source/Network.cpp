#include "Network.h"	


Network::Network(int _datapointSize, int _labelSize) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{
	output = new float[labelSize];

	nodes.resize(datapointSize + labelSize);
	for (int i = 0; i < nodes.size(); i++) 
	{
		nodes[i] = new Node(0, nullptr);
	}
}

Network::~Network() 
{
	delete[] output;
	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}
}


void Network::asynchronousLearn(float* _datapoint, float* _label, int nSteps)
{
	setDatapoint(_datapoint);
	setLabel(_label);

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		LOG(previousEnergy);

		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->asynchronousActivationGradientStep();
			nodes[i]->asynchronousWeightGradientStep();
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy);	

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->learnIncomingXWBvariates();
	}

	LOG(" WBU  " << computeTotalActivationEnergy());
	LOGL("\n");
}

void Network::asynchronousEvaluate(float* _datapoint, int nSteps) 
{
	setDatapoint(_datapoint);

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->asynchronousActivationGradientStep();
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}

	LOG(previousEnergy);

	for (int i = 0; i < labelSize; i++)
	{
		output[i] = nodes[datapointSize + i]->x;
	}
}



float Network::computeTotalActivationEnergy()
{
	float E = 0.f;

	for (int i = 0; i < nodes.size(); i++)
	{
		E += nodes[i]->epsilon * nodes[i]->epsilon * nodes[i]->tau - logf(nodes[i]->tau); // + constants
	}

	return .5f * E;
}


void Network::setDatapoint(float* _datapoint)
{
	for (int i = 0; i < datapointSize; i++)
	{
		nodes[i]->setActivation(_datapoint[i]);
	}
}

void Network::setLabel(float* _label) 
{
	for (int i = 0; i < labelSize; i++)
	{
		nodes[i + datapointSize]->setActivation(_label[i]);
	}
}


void Network::initializeEpsilons() 
{
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->computeEpsilon();
	}
}

