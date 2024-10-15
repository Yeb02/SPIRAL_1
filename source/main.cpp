#include "Network.h"
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
	bool useMNIST = false;
	bool testRetrocausal = true;
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

		auto [trainShuffledPoints, trainShuffledLabels] = create_batches(trainDatapoints, trainLabels, trainSetSize);
		auto [testShuffledPoints, testShuffledLabels] = create_batches(testDatapoints, testLabels, testSetSize);
	}
	else if (testRetrocausal)
	{
		datapointS = 1;
		labelS = 2;

		testSetSize = 10000;
		trainSetSize = 10000;


		int s = datapointS + labelS;

		trainShuffledPoints = new float* [trainSetSize];
		trainShuffledLabels = new float* [trainSetSize];
		float* trainData = new float[s * trainSetSize];
		for (int i = 0; i < trainSetSize; i++) 
		{
			trainData[s * i] = (UNIFORM_01 > .5f) ? 1.f : -1.f;
			trainData[s * i + 1] = (trainData[s * i] == -1.f) ? 1.f : ((UNIFORM_01 > .5f) ? 1.f : -1.f);
			trainData[s * i + 2] = (trainData[s * i] == 1.f) ? 1.f : ((UNIFORM_01 > .5f) ? 1.f : -1.f);

			trainShuffledPoints[i] = trainData + s * i;
			trainShuffledLabels[i] = trainData + s * i + datapointS;
		}

		testShuffledPoints = new float* [testSetSize];
		testShuffledLabels = new float* [testSetSize];
		float* testData = new float[s * testSetSize];
		for (int i = 0; i < testSetSize; i++)
		{
			testData[s * i] = (UNIFORM_01 > .5f) ? 1.f : -1.f;
			testData[s * i + 1] = (testData[s * i] == -1.f) ? 1.f : ((UNIFORM_01 > .5f) ? 1.f : -1.f);
			testData[s * i + 2] = (testData[s * i] == 1.f) ? 1.f : ((UNIFORM_01 > .5f) ? 1.f : -1.f);

			testShuffledPoints[i] = testData + s * i;
			testShuffledLabels[i] = testData + s * i + datapointS;
		}
	}



	Node::xlr = .1f;

	Node::wxPriorStrength = 1.0f;
	Node::wtPriorStrength = 1.0f;

	Node::observationImportance = 1.0f;
	Node::certaintyDecay = .01f;

	Node::energyDecay = .025f;
	Node::connexionEnergyThreshold = 10.f;
		
	Node::xReg  = .05f;   //0.15f taus fixés topo fixée   // TODO remove xreg from observation nodes
	Node::wxReg = .001f; //0.05f taus fixés topo fixée
	Node::wtReg = .05f;

	
	constexpr bool dynamicTopology = false; 
#ifdef DYNAMIC_PRECISIONS // TODO check that good values for these parameters still vary wildly if DYNAMIC_PRECISIONS is switched
	Network::KC = 4.f;
	Network::KN = 50.f;
#else
	Network::KC = 12.f;
	Network::KN = 100.f;
#endif

	// C++ is really stupid sometimes
	const int _nLayers = 4;
	int _sizes[_nLayers + 2] = {0, datapointS + labelS, 20, 15, 10, 0};
	/*const int _nLayers = 2;
	int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 10, 0 };*/


	int nLayers = _nLayers;
	int* sizes = &(_sizes[1]);
	if (dynamicTopology) 
	{
		nLayers = 0;
		sizes = nullptr;
	}
		
	Network nn(datapointS, labelS, nLayers, sizes);

	int nTrainSteps = 10; // Suprisingly, less steps leads to much better results. More steps requires lower wxlr.
	int nTestSteps = 10;
	int nTests = 1000;
	
	// one and only one must be set to  true
	bool onePerClass = false;
	bool onlineRandom = true;
	bool timeDependancy = false;
	if (onePerClass) {
		for (int u = 0; u < 5; u++) {
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
		for (int i = 0; i < 1000; i++)
		{
			nn.learn(trainShuffledPoints[i], trainShuffledLabels[i], nTrainSteps);
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

	if (timeDependancy) {
		int nCorrects = 0;
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
			nCorrects += isCorrect;


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
			nCorrects += isCorrect;
			id++;
		}

		LOGL("\n" << (float)nCorrects / (float)nTests);
	}
	else if (testRetrocausal) 
	{
		float* output = nn.output;
		float avgMSE = .0f;
		for (int i = 0; i < 1000; i++)
		{
			nn.evaluate(testShuffledPoints[i], nTestSteps);
			float MSE_loss = .0f;
			for (int j = 0; j < labelS; j++)
			{
				MSE_loss += powf(output[j] - testShuffledLabels[i][j], 2.0f);
			}
			LOG("\n");
			LOG(testShuffledLabels[i][0]);
			LOG(testShuffledLabels[i][1]);
			LOG(testShuffledLabels[i][2]);
			LOGL("\n" + std::to_string(MSE_loss) + "\n");
			avgMSE += MSE_loss;
		}
		LOGL("\n\n\n" + std::to_string(avgMSE / (float)1000));
	}
	else {
		int nCorrects = 0;
		float* output = nn.output;
		for (int i = 0; i < nTests; i++)
		{
			nn.evaluate(testShuffledPoints[i], nTestSteps);

			LOG("\n");


			float MSE_loss = .0f;
			for (int j = 0; j < labelS; j++)
			{
				MSE_loss += powf(output[j] - testShuffledLabels[i][j], 2.0f);
			}
			int isCorrect = isCorrectAnswer(output, testShuffledLabels[i]);
			LOGL(isCorrect << " " << MSE_loss << "\n");
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

