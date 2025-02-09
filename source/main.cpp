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
	bool testRetrocausal = !useMNIST;
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





	Node::xlr = .7f; 
	Node::wxPriorStrength = 1.f;
	Node::observationImportance = 1.f;
	Node::certaintyDecay = .001f;
	Node::xReg  = .2f;  
	Node::wxReg = .0f;  

	int nTrainSteps = 5; // Suprisingly, less steps leads to much better results.
	int nTestSteps = 5;


#ifdef VANILLA_PREDICTIVE_CODING
	Node::xlr = .1f;
	Node::xReg = .0f;  
	Node::wxReg = .0f; 
	nTrainSteps = 10; 
	nTestSteps = 10;
#endif

	Network nn(datapointS, labelS); // datapoint is group 0, label is group 1.
	int topo = 0;

	switch (topo) {
		case 0: {
			nn.addGroup(50); //group 2
			nn.addGroup(30); //group 3
			nn.addGroup(15); //group 4
			nn.addConnexion(4, 3);
			nn.addConnexion(3, 2);
			nn.addConnexion(2, 1);
			nn.addConnexion(2, 0);
			//nn.addConnexion(1, 1);
			//nn.addConnexion(1, 2);
			//nn.addConnexion(2, 4);
			break;
		}
		case 1: {
			nn.addGroup(20); //group 2
			//nn.addGroup(5);  //group 3
			//nn.addConnexion(3, 2);
			//nn.addConnexion(3, 3);
			//nn.addConnexion(0, 2);
			nn.addConnexion(2, 0);
			//nn.addConnexion(2, 2);

			//nn.addConnexion(1, 2);
			nn.addConnexion(2, 1);
			break;
		}
		case 2: {
			nn.addGroup(1);  //group 2
			nn.addConnexion(2, 1);
			nn.addConnexion(2, 0);
			break;
		}
		case 3: { // VANILLA_PREDICTIVE_CODING  configuration: top down predictions, label -> hidden layers -> observations
			nn.addGroup(50);  //group 2
			nn.addConnexion(2, 0);
			nn.addConnexion(1, 2);
			break;
		}
	}
	nn.initialize();



	// ANetwork
	
	//ANode::wReg = .25f;
	//ANode::wPriorStrength = .02f;
	//ANode::observationImportance = .02f;
	//ANode::certaintyDecay = .01f;
	//ANode::xReg = .1f;
	//ANetwork nn(datapointS, labelS);
	//{
	//	float target_density = .1f;
	//	float density_strength = 2.f;
	//	float target_freqency = .2f;
	//	float freqency_strength = 1.f;
	//	int nNodes = 300;
	//	Assembly* a2 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
	//	nn.addAssembly(a2);
	//	//Assembly* a3 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
	//	//nn.addAssembly(a3);
	//	//Assembly* a4 = new Assembly(nNodes, target_density, density_strength, target_freqency, freqency_strength);
	//	//nn.addAssembly(a4);
	//	//nn.addConnexion(2, 0, .2f);
	//	//nn.addConnexion(2, 1, 1.f);
	//	//nn.addConnexion(2, 2, 1.f);
	//	//nn.addConnexion(2, 0, 1.f);
	//	//nn.addConnexion(2, 1, 1.f);
	//	//nn.addConnexion(2, 2, 1.f);
	//	float i_f = 1.f;
	//	float o_f = 1.f;
	//	nn.addConnexion(2, 0, 1.f);
	//	nn.addConnexion(2, 1, o_f);
	//	//nn.addConnexion(3, 2, 1.f);
	//	//nn.addConnexion(3, 1, o_f);
	//	//nn.addConnexion(3, 3, i_f);
	//	//nn.addConnexion(4, 3, o_f);
	//	//nn.addConnexion(4, 1, 1.f);
	//	//nn.addConnexion(4, 4, i_f);
	//}
	//int nTrainSteps = 4;
	//int nTestSteps = 4;





	nn.readyForLearning();

	// one and only one must be set to  true
	bool onePerClass = false;
	bool onlineRandom = !false;
	bool timeDependancy = false;
	if (onePerClass) {
		for (int u = 0; u < 2; u++) {
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
			//Node::xlr = std::min(.7f, .1f + .6f * (float)i / 50);
			//nTrainSteps = std::max(5, 15 - i / 3);
			nn.learn(trainShuffledPoints[i], trainShuffledLabels[i], nTrainSteps);
			if (i % 100 == 99) {
				LOGL("Step " + std::to_string(i));
			}
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
}

