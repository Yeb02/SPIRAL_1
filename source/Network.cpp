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

	initializeEpsilons();

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		LOG(previousEnergy);

		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->asynchronousActivationGradientStep();
			//nodes[i]->updateIncomingXWBvariates(); // ? TODO
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy);	

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->updateIncomingXWBvariates();
		nodes[i]->learnIncomingXWBvariates();
	}

	LOG(" WBU  " << computeTotalActivationEnergy());
	LOGL("\n");
}

void Network::asynchronousEvaluate(float* _datapoint, int nSteps) 
{
	setDatapoint(_datapoint);

	initializeEpsilons();

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
		nodes[i]->x = _datapoint[i];
		nodes[i]->fx = tanhf(_datapoint[i]); // Not needed right now, but could be forgotten in future versions...
	}
}

void Network::setLabel(float* _label) 
{
	for (int i = 0; i < labelSize; i++)
	{
		nodes[i + datapointSize]->x = _label[i];
		nodes[i + datapointSize]->fx = tanhf(_label[i]); // Not needed right now, but could be forgotten in future versions...
	}
}


void Network::initializeEpsilons() 
{
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->computeEpsilon();
	}
}

