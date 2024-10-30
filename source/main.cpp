#include "Network.h"
#include "ANetwork.h"
#include "MNIST.h"



int main()
{
	LOG(std::setprecision(4));

	int datapointS, labelS;
	int trainSetSize;
	float** trainShuffledPoints = nullptr;
	float** trainShuffledLabels = nullptr;
	int testSetSize;
	float** testShuffledPoints = nullptr;
	float** testShuffledLabels = nullptr;

	// one and only one must be set to  true
	bool useMNIST = true;
	bool testRetrocausal = false;
	if (useMNIST)
	{
		datapointS = 28 * 28;
		labelS = 10;

		testSetSize = 10000;
		float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", testSetSize);
		float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", testSetSize);

		trainSetSize = 60000;
		float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", trainSetSize);
		float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", trainSetSize);

		auto [a1, a2] = create_batches(trainDatapoints, trainLabels, trainSetSize);
		trainShuffledPoints = a1;
		trainShuffledLabels = a2;
		auto [b1, b2] = create_batches(testDatapoints, testLabels, testSetSize);
		testShuffledPoints = b1;
		testShuffledLabels = b2;
	}
	else if (testRetrocausal)
	{
		// when datapoint=1 is observed, the most likely underlying cause is c=1, which is undecisive regarding the label. Therefore it is a better idea to
		// infer c=0 that lets us conclude on the value of D
		datapointS = 1; 

		labelS = 1;

		testSetSize = 10000;
		trainSetSize = 10000;


		int s = datapointS + labelS;
		const float t=.5f, u=.5f, v=.7f, a=.55f, b=.95f; 

		trainShuffledPoints = new float* [trainSetSize];
		trainShuffledLabels = new float* [trainSetSize];
		float* trainData = new float[s * trainSetSize];
		for (int i = 0; i < trainSetSize; i++) 
		{
			float c = (UNIFORM_01 < t) ? 0.f : 1.f; // p(C=0) = t
			trainData[s * i] = (UNIFORM_01 < (c*v + (1.f-c)*u)) ? 1.f : 0.f; // p(G=1|C=0) = u, p(G=1|C=1) = v
			trainData[s * i + 1] = (UNIFORM_01 < (c*a + (1.f-c)*b)) ? 1.f : 0.f; // p(D=1|C=0) = b, p(D=1|C=1) = a
	

			trainShuffledPoints[i] = trainData + s * i;
			trainShuffledLabels[i] = trainData + s * i + datapointS;
		}

		testShuffledPoints = new float* [testSetSize];
		testShuffledLabels = new float* [testSetSize];
		float* testData = new float[s * testSetSize];
		for (int i = 0; i < testSetSize; i++)
		{
			float c = (UNIFORM_01 < t) ? 0.f : 1.f; // p(C=0) = t
			testData[s * i] = (UNIFORM_01 < (c * v + (1.f - c) * u)) ? 1.f : 0.f; // p(G=1|C=0) = u, p(G=1|C=1) = v
			testData[s * i + 1] = (UNIFORM_01 < (c * a + (1.f - c) * b)) ? 1.f : 0.f; // p(D=1|C=0) = b, p(D=1|C=1) = a

			testShuffledPoints[i] = testData + s * i;
			testShuffledLabels[i] = testData + s * i + datapointS;
		}
	}



	constexpr bool dynamicTopology = false;

//	Node::xlr = 1.f;
//	Node::wxPriorStrength = 1.0f;
//	Node::wtPriorStrength = 1.0f;
//	Node::observationImportance = 1.0f;
//	Node::certaintyDecay = .01f;
//	Node::energyDecay = .01f;
//	Node::connexionEnergyThreshold = 1.f;
//	Node::xReg  = .1f;   
//	Node::wxReg = .05f;  
//	Node::wtReg = .05f;
//
//	int nTrainSteps = 4; // Suprisingly, less steps leads to much better results. More steps requires lower wxlr.
//	int nTestSteps = 4;
//
//#ifdef DYNAMIC_PRECISIONS // TODO check that good values for these parameters still vary wildly if DYNAMIC_PRECISIONS is switched
//	Network::KC = 4.f;
//	Network::KN = 50.f;
//#else
//	Network::KC = 15.f;
//	Network::KN = 500.f;
//#endif
//
//#ifdef VANILLA_PREDICTIVE_CODING
//	Node::xlr = .1f;
//	Node::xReg = .0f;  
//	Node::wxReg = .00f; 
//	nTrainSteps = 10; 
//	nTestSteps = 10;
//#endif
//
//	// C++ is really stupid sometimes
//	/*const int _nLayers = 5;
//	int _sizes[_nLayers + 2] = {0, datapointS + labelS, 50, 25, 15, 5, 0};*/
//	const int _nLayers = 2;
//	int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 30, 0 };
//	//const int _nLayers = 2;
//	//int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 3, 0 };
//
//	int nLayers = _nLayers;
//	int* sizes = &(_sizes[1]);
//	if (dynamicTopology) 
//	{
//		nLayers = 0;
//		sizes = nullptr;
//	}
//	Network nn(datapointS, labelS, nLayers, sizes);




	ANode::wReg = .25f;
	ANode::wPriorStrength = .1f;
	ANode::observationImportance = .1f;
	ANode::certaintyDecay = .01f;
	ANode::xReg = .1f;

	ANetwork nn(datapointS, labelS);
	if (true)
	{
		float target_density = .5f;
		float density_strength = .1f;
		float target_freqency = .5f;
		float freqency_strength = .1f;
		int nNodes = 200;
		Assembly* a2 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
		nn.addAssembly(a2);
		Assembly* a3 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
		nn.addAssembly(a3);
		//Assembly* a4 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
		//nn.addAssembly(a4);

		//nn.addConnexion(2, 0, .2f);
		//nn.addConnexion(2, 1, 1.f);
		//nn.addConnexion(2, 2, 1.f);

		//nn.addConnexion(2, 0, 1.f);
		//nn.addConnexion(2, 1, 1.f);
		//nn.addConnexion(2, 2, 1.f);

		float i_f = 1.f;
		float o_f = 1.f;
		nn.addConnexion(2, 0, 1.f);
		//nn.addConnexion(2, 2, i_f);
		nn.addConnexion(3, 2, o_f);
		nn.addConnexion(3, 1, o_f);
		//nn.addConnexion(3, 3, i_f);
		//nn.addConnexion(4, 3, o_f);
		//nn.addConnexion(4, 1, 1.f);
		//nn.addConnexion(4, 4, i_f);
	}
	int nTrainSteps = 3;
	int nTestSteps = 3;





	nn.readyForLearning();

	// one and only one must be set to  true
	bool onePerClass = false;
	bool onlineRandom = true;
	bool timeDependancy = false;
	if (onePerClass) {
		for (int u = 0; u < 1; u++) {
			for (int i = 0; i < 10; i++) {
				int id = 0;
				while (trainShuffledLabels[id][i] != 1.0f) {
					id++;
				}
				nn.learn(trainShuffledPoints[id], trainShuffledLabels[id], nTrainSteps);
			}
			LOG("LOOP " << u << " done.\n\n")
		}
	}
	else if (onlineRandom){
		for (int i = 0; i < 100; i++)
		{
			nn.learn(trainShuffledPoints[i], trainShuffledLabels[i], nTrainSteps);
			if (i % 100 == 99) LOGL("Step " + std::to_string(i));
		}
	}
	else if (timeDependancy){
		int id = 0;
		for (int u = 0; u < 250; u++) { // 50k train datapoints so (250 * 10) < 50000 is safe in expectation
			nn.learn(trainShuffledPoints[id], trainShuffledLabels[id], nTrainSteps);

			int label = -1;
			for (int i = 0; i < 10; i++)
			{
				if (trainShuffledLabels[id][i] == 1.0f) {
					label = i;
					break;
				}
			}

			int j = 1;
			while (trainShuffledLabels[id+j][(label+1)%10] != 1.0f) {
				j++;
			}
				
			id = id + j;
			nn.learn(trainShuffledPoints[id], trainShuffledLabels[id], nTrainSteps);
			id++;
		}
	}

	if (dynamicTopology)
	{
		for (int j = labelS+datapointS; j < nn.getNNodes(); j++)
		{
			LOGL("p " << nn.nodes[j]->parents.size() << "   c " << nn.nodes[j]->children.size());
		}
	}

	nn.readyForTesting();

	int nTests = 1000;
	if (timeDependancy) {
		int n1Corrects = 0;
		int n2Corrects = 0;
		float* output = nn.output;
		int id = 0;
		for (int u = 0; u < 500; u++) { // 10k test datapoints so (500 * 10) < 10000 is safe in expectation
				
			nn.evaluate(testShuffledPoints[id], nTestSteps);
			LOG("\n");
			float MSE_loss = .0f;
			for (int v = 0; v < labelS; v++)
			{
				MSE_loss += powf(output[v] - testShuffledLabels[id][v], 2.0f);
			}
			int isCorrect = isCorrectAnswer(output, testShuffledLabels[id]);
			LOGL(isCorrect << " " << MSE_loss << "\n");
			n1Corrects += isCorrect;


			int label = -1;
			for (int i = 0; i < 10; i++)
			{
				if (testShuffledLabels[id][i] == 1.0f) {
					label = i;
					break;
				}
			}

			int j = 1;
			while (testShuffledLabels[id + j][(label + 1) % 10] != 1.0f) {
				j++;
			}

			id = id + j;
			nn.evaluate(testShuffledPoints[id], nTestSteps);
			LOG("\n");
			MSE_loss = .0f;
			for (int v = 0; v < labelS; v++)
			{
				MSE_loss += powf(output[v] - testShuffledLabels[id][v], 2.0f);
			}
			isCorrect = isCorrectAnswer(output, testShuffledLabels[id]);
			LOGL(isCorrect << " " << MSE_loss << "\n");
			n2Corrects += isCorrect;
			id++;
		}

		LOGL("\nRdm: " << 2.f * (float)n1Corrects / (float)nTests << ", next: " << 2.f * (float)n2Corrects / (float)nTests);
		LOGL((float)(n1Corrects + n2Corrects) / (float)nTests);
	}
	else if (testRetrocausal) 
	{
		nTests = 100;
		float* output = nn.output;
		float avgMSE = .0f;
		for (int i = 0; i < nTests; i++)
		{
			nn.evaluate(testShuffledPoints[i], nTestSteps);
			LOG("\n");
			for (int j = 0; j < datapointS; j++) LOG(testShuffledPoints[i][j]);
			for (int j = 0; j < labelS; j++) LOG(testShuffledLabels[i][j]);
			LOG("\n");
			for (int j = 0; j < labelS; j++) LOG(output[j]);
			LOGL("\n");
		}
	}
	else {

		int nCorrects = 0;
		float* output = nn.output;
		for (int i = 0; i < nTests; i++)
		{
			nn.evaluate(testShuffledPoints[i], nTestSteps);

			//LOG("\n");


			float MSE_loss = .0f;
			for (int j = 0; j < labelS; j++)
			{
				MSE_loss += powf(output[j] - testShuffledLabels[i][j], 2.0f);
			}
			int isCorrect = isCorrectAnswer(output, testShuffledLabels[i]);
			//LOGL(isCorrect << " " << MSE_loss << "\n");
			nCorrects += isCorrect;
		}
		LOGL("\n" << (float)nCorrects / (float)nTests);
	}
		

	if (dynamicTopology)
	{
		int nAddedNodes = nn.getNNodes() - labelS - datapointS;
		LOGL("\nNetwork has added " + std::to_string(nAddedNodes) + " new nodes.\n");
	}

}

