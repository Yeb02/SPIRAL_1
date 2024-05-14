#include "FCN.h"
#include "MNIST.h"


int isCorrectAnswer(float* out, float* y) {
	float outMax = -1000.0f;
	int imax = -1;
	for (int i = 0; i < 10; i++)
	{
		if (outMax < out[i]) {
			outMax = out[i];
			imax = i;
		}
	}

	return y[imax] == 1.0f ? 1 : 0;
}

int main()
{

	const int datapointS = 28 * 28;
	const int labelS = 10;

	const int testSetSize = 10000;
	float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", testSetSize);
	float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", testSetSize);

	const int trainSetSize = 60000;
	float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", trainSetSize);
	float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", trainSetSize);

	int batchSize = 1;
	auto [batchedPoints, batchedLabels] = create_batches(trainDatapoints, trainLabels, trainSetSize, batchSize);
	auto [testBatchedPoints, testBatchedLabels] = create_batches(testDatapoints, testLabels, testSetSize, batchSize);



	int nThreads = std::thread::hardware_concurrency();
	LOGL(nThreads << " concurrent threads are supported at hardware level.");
	//nThreads -= 1; // if your computer lags, uncomment this.

	//nThreads = 1;

	LOGL("Using " << nThreads << ".\n");



#ifdef LABEL_IS_DATAPOINT
	const int nLayers = 2;
	int sizes[nLayers] = { datapointS + labelS, 10 };
#else
	const int nLayers = 2;
	int sizes[nLayers] = { datapointS, labelS };
#endif
	
	
	float Kw = .002f;
	float stepSize = .01f;


	FCN nn(nLayers, &sizes[0], Kw, stepSize, nThreads, datapointS);



	// poor, TODO
#ifdef LABEL_IS_DATAPOINT
	float* output = &nn.x[0][datapointS];
#else
	float* output = nn.x[nLayers - 1];
#endif


	for (int u = 0; u < 10; u++) {
		for (int i = 0; i < 10; i++) {
			int id = 0;
			while (batchedLabels[id][i] != 1.0f) {
				id++;
			}
			nn.learn(batchedPoints[id], batchedLabels[id], 15);
		}
	}

	//for (int i = 0; i < 100; i++)
	//{
	//	nn.learn(batchedPoints[i], batchedLabels[i], 15);
	//}


	//nn.stepSize = .01f;
	int nTests = 1000;
	int nCorrects = 0;
	for (int i = 0; i < nTests; i++)
	{
		nn.evaluate(testBatchedPoints[i], 20);

		float MSE_loss = .0f;
		for (int j = 0; j < labelS; j++)
		{
			MSE_loss += powf(output[j] - testBatchedLabels[i][j], 2.0f);
		}
		int isCorrect = isCorrectAnswer(output, testBatchedLabels[i]);
		LOGL(isCorrect << " " << MSE_loss);
		nCorrects += isCorrect;
	}
	LOGL("\n" << (float) nCorrects / (float) nTests);

}

