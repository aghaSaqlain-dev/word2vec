#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <limits>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <cfloat>

using namespace std;

// Cosine similarity between two vectors
double cosineSimilarity(const vector<double> &vec1, const vector<double> &vec2)
{
    double dotProduct = 0.0, norm1 = 0.0, norm2 = 0.0;
    
    // Make sure vectors are same length
    if (vec1.size() != vec2.size()) {
        cerr << "Error: Vector dimensions don't match! " << vec1.size() << " vs " << vec2.size() << endl;
        return 0.0;
    }
    
    for (size_t i = 0; i < vec1.size(); i++)
    {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    // Add safety check for division by zero
    if (norm1 < 1e-10 || norm2 < 1e-10)
        return 0.0;
        
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}

// Find the most similar word from the context matrix
string findMostSimilarWord(const vector<double> &wordEmbedding,
                           const vector<vector<double>> &contextMatrix,
                           const vector<string> &vocabulary)
{
    double maxSimilarity = -1.0;
    size_t bestWordIndex = 0;
    for (size_t i = 0; i < vocabulary.size(); i++)
    {
        vector<double> contextVector(contextMatrix.size());
        for (size_t j = 0; j < contextMatrix.size(); j++)
            contextVector[j] = contextMatrix[j][i];
        double similarity = cosineSimilarity(wordEmbedding, contextVector);
        if (similarity > maxSimilarity)
        {
            maxSimilarity = similarity;
            bestWordIndex = i;
        }
    }
    return vocabulary[bestWordIndex];
}

// Efficiently build vocabulary from words
vector<string> buildVocabulary(const vector<string> &words)
{
    unordered_set<string> vocabSet(words.begin(), words.end());
    return vector<string>(vocabSet.begin(), vocabSet.end());
}

// Compute word frequencies for subsampling
unordered_map<string, double> computeWordFrequencies(const vector<string> &words)
{
    unordered_map<string, int> wordCounts;
    for (const auto &word : words)
        wordCounts[word]++;
    
    unordered_map<string, double> wordFreqs;
    for (const auto &pair : wordCounts)
        wordFreqs[pair.first] = static_cast<double>(pair.second) / words.size();
    
    return wordFreqs;
}

// Calculate sub-sampling probability for a word
double getSubsamplingProb(double freq, double threshold = 1e-5)
{
    if (freq < threshold)
        return 1.0;
    return (sqrt(freq / threshold) + 1) * threshold / freq;
}

// Helper to get a context vector from context matrix
vector<double> getContextVector(const vector<vector<double>> &contextMatrix, size_t wordIndex)
{
    vector<double> contextVector(contextMatrix.size());
    for (size_t j = 0; j < contextMatrix.size(); j++)
        contextVector[j] = contextMatrix[j][wordIndex];
    return contextVector;
}

// Sigmoid function for negative sampling
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Print matrix with vocabulary labels
void printMatrix(const vector<vector<double>> &matrix, const vector<string> &vocabulary, const string &title)
{
    cout << title << endl;
    for (size_t i = 0; i < matrix.size(); i++)
    {
        if (!vocabulary.empty())
            cout << vocabulary[i] << ": ";
        for (const auto &val : matrix[i])
            cout << val << " ";
        cout << endl;
    }
    cout << "------------------------" << endl;
}

// Training loop with all Word2Vec features
void trainModel(
    vector<vector<double>> &embeddingMatrix,
    vector<vector<double>> &contextMatrix,
    const vector<string> &words,
    const vector<string> &vocabulary,
    const unordered_map<string, size_t> &wordToIndex,
    const unordered_map<string, double> &wordFrequencies,
    int numEpochs,
    double initialLearningRate,
    int windowSize = 2,
    int negSamples = 5,
    double subsampleThreshold = 1e-5)
{
    // Create noise distribution for negative sampling (based on word frequencies^0.75)
    vector<double> noiseDist(vocabulary.size());
    double normalizationFactor = 0.0;
    for (size_t i = 0; i < vocabulary.size(); i++) {
        double freq = wordFrequencies.at(vocabulary[i]);
        noiseDist[i] = pow(freq, 0.75);
        normalizationFactor += noiseDist[i];
    }
    for (size_t i = 0; i < noiseDist.size(); i++) {
        noiseDist[i] /= normalizationFactor;
    }

    // Setup random generators for each thread
    vector<mt19937> randomEngines;
    vector<discrete_distribution<int>> negSamplers;
    
    int nThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nThreads = omp_get_num_threads();
    }
    
    for (int t = 0; t < nThreads; t++) {
        unsigned int seed = static_cast<unsigned int>(time(nullptr)) + t;
        randomEngines.emplace_back(seed);
        negSamplers.emplace_back(noiseDist.begin(), noiseDist.end());
    }
    
    // Epoch loop with adaptive learning rate
    double prevLoss = DBL_MAX;
    const double earlyStoppingThreshold = 0.001;
    
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        double totalLoss = 0.0;
        
        // Adaptive learning rate (linear decay)
        double learningRate = initialLearningRate * (1.0 - static_cast<double>(epoch) / numEpochs);
        
        // Global gradients for model update
        vector<vector<double>> contextMatrixGrad(contextMatrix.size(), vector<double>(contextMatrix[0].size(), 0.0));
        vector<vector<double>> embeddingMatrixGrad(embeddingMatrix.size(), vector<double>(embeddingMatrix[0].size(), 0.0));
        
        // Thread-local gradient buffers
        vector<vector<vector<double>>> threadContextGrads(nThreads, vector<vector<double>>(contextMatrix.size(), vector<double>(contextMatrix[0].size(), 0.0)));
        vector<vector<vector<double>>> threadEmbeddingGrads(nThreads, vector<vector<double>>(embeddingMatrix.size(), vector<double>(embeddingMatrix[0].size(), 0.0)));
        vector<double> threadLosses(nThreads, 0.0);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &localContextGrad = threadContextGrads[tid];
            auto &localEmbeddingGrad = threadEmbeddingGrads[tid];
            double &localLoss = threadLosses[tid];
            
            // Local RNG
            auto &rng = randomEngines[tid];
            auto &negativeSampler = negSamplers[tid];
            uniform_real_distribution<double> uniform(0.0, 1.0);
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < words.size(); ++i)
            {
                // Skip words based on subsampling probability
                auto wordIt = wordToIndex.find(words[i]);
                if (wordIt == wordToIndex.end())
                    continue;
                    
                size_t currentWordIndex = wordIt->second;
                double freq = wordFrequencies.at(words[i]);
                double keepProb = getSubsamplingProb(freq, subsampleThreshold);
                
                if (uniform(rng) > keepProb)
                    continue; // Skip this word
                
                // Process window context words
                vector<double> embeddingResult = embeddingMatrix[currentWordIndex];
                
                // Iterate over context window
                for (int j = -windowSize; j <= windowSize; j++)
                {
                    if (j == 0) continue; // Skip the central word
                    
                    size_t contextPos = i + j;
                    if (contextPos >= words.size())
                        continue;
                        
                    auto targetIt = wordToIndex.find(words[contextPos]);
                    if (targetIt == wordToIndex.end())
                        continue;
                        
                    size_t targetWordIndex = targetIt->second;
                    
                    // NEGATIVE SAMPLING: Train on target word (positive) and negative samples
                    
                    // Process positive example (target word)
                    vector<double> contextVector = getContextVector(contextMatrix, targetWordIndex);
                    double dotProduct = 0.0;
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        dotProduct += embeddingResult[d] * contextVector[d];
                    
                    double sigmoid_pos = sigmoid(dotProduct);
                    double loss_pos = -log(sigmoid_pos + 1e-10);
                    localLoss += loss_pos;
                    
                    // Update gradients for positive example
                    double grad_pos = (sigmoid_pos - 1.0); // derivative of -log(sigmoid(x))
                    
                    // Update context matrix gradient for positive sample
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        localContextGrad[d][targetWordIndex] += grad_pos * embeddingResult[d];
                        
                    // Update embedding matrix gradient
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        localEmbeddingGrad[currentWordIndex][d] += grad_pos * contextVector[d];
                    
                    // Process negative examples
                    for (int neg = 0; neg < negSamples; neg++)
                    {
                        // Sample a random negative word
                        size_t negWordIndex = negativeSampler(rng);
                        
                        // Skip if we accidentally sample the target word
                        if (negWordIndex == targetWordIndex)
                            continue;
                            
                        vector<double> negContextVector = getContextVector(contextMatrix, negWordIndex);
                        double negDotProduct = 0.0;
                        for (size_t d = 0; d < embeddingResult.size(); d++)
                            negDotProduct += embeddingResult[d] * negContextVector[d];
                            
                        double sigmoid_neg = sigmoid(negDotProduct);
                        double loss_neg = -log(1.0 - sigmoid_neg + 1e-10);
                        localLoss += loss_neg;
                        
                        // Update gradients for negative example
                        double grad_neg = sigmoid_neg; // derivative of -log(1-sigmoid(x))
                        
                        // Update context matrix gradient for negative sample
                        for (size_t d = 0; d < embeddingResult.size(); d++)
                            localContextGrad[d][negWordIndex] += grad_neg * embeddingResult[d];
                            
                        // Update embedding matrix gradient
                        for (size_t d = 0; d < embeddingResult.size(); d++)
                            localEmbeddingGrad[currentWordIndex][d] += grad_neg * negContextVector[d];
                    }
                }
            }
        }
        
        // Sum thread-local gradients into global gradients
        for (int t = 0; t < nThreads; ++t)
        {
            for (size_t d = 0; d < contextMatrix.size(); d++)
                for (size_t w = 0; w < contextMatrix[0].size(); w++)
                    contextMatrixGrad[d][w] += threadContextGrads[t][d][w];

            for (size_t i = 0; i < embeddingMatrix.size(); i++)
                for (size_t d = 0; d < embeddingMatrix[0].size(); d++)
                    embeddingMatrixGrad[i][d] += threadEmbeddingGrads[t][i][d];

            totalLoss += threadLosses[t];
        }
        
        // Update weights with adaptive learning rate
        for (size_t d = 0; d < contextMatrix.size(); d++)
            for (size_t w = 0; w < contextMatrix[0].size(); w++)
                contextMatrix[d][w] -= learningRate * contextMatrixGrad[d][w];

        for (size_t i = 0; i < embeddingMatrix.size(); i++)
            for (size_t d = 0; d < embeddingMatrix[0].size(); d++)
                embeddingMatrix[i][d] -= learningRate * embeddingMatrixGrad[i][d];
                
        if (epoch % 10 == 0 || epoch == numEpochs - 1)
            cout << "Epoch " << epoch << ", Learning Rate: " << learningRate 
                 << ", Average Loss: " << totalLoss / words.size() << endl;
                 
        // Early stopping
        if (epoch > 10 && abs(prevLoss - totalLoss) < earlyStoppingThreshold * prevLoss) {
            cout << "Early stopping at epoch " << epoch << endl;
            break;
        }
        prevLoss = totalLoss;
    }
}

// Interactive prediction loop
void interactivePrediction(
    const vector<vector<double>> &embeddingMatrix,
    const vector<vector<double>> &contextMatrix,
    const vector<string> &vocabulary,
    const unordered_map<string, size_t> &wordToIndex)
{
    cout << "Word Prediction Mode (enter 'exit' to quit)" << endl;
    cout << "------------------------" << endl;
    
    cout << "Embedding dimension: " << embeddingMatrix[0].size() << endl;
    cout << "Context matrix dimensions: " << contextMatrix.size() << " x " << contextMatrix[0].size() << endl;
    
    string inputWord;
    while (true)
    {
        cout << "Enter a word: ";
        cin >> inputWord;
        if (inputWord == "exit")
            break;
        auto wordIt = wordToIndex.find(inputWord);
        if (wordIt == wordToIndex.end())
        {
            cout << "Word not in vocabulary. Please try another word." << endl;
            continue;
        }
        size_t wordIndex = wordIt->second;
        vector<double> wordEmbedding = embeddingMatrix[wordIndex];

        // Compute similarities for all words
        vector<pair<string, double>> similarities;
        for (size_t i = 0; i < vocabulary.size(); i++)
        {
            vector<double> contextVector(contextMatrix.size());
            for (size_t j = 0; j < contextMatrix.size(); j++) {
                if (i < contextMatrix[j].size()) { // Safety check
                    contextVector[j] = contextMatrix[j][i];
                } else {
                    cerr << "Index out of bounds: " << i << " >= " << contextMatrix[j].size() << endl;
                    contextVector[j] = 0.0;
                }
            }
            double similarity = cosineSimilarity(wordEmbedding, contextVector);
            similarities.emplace_back(vocabulary[i], similarity);
        }

        // Sort by similarity in descending order
        sort(similarities.begin(), similarities.end(),
             [](const pair<string, double> &a, const pair<string, double> &b)
             {
                 return a.second > b.second;
             });

        // Output most similar words
        cout << "Top 10 most similar words to '" << inputWord << "':" << endl;
        for (size_t i = 0; i < min(size_t(10), similarities.size()); i++) {
            cout << similarities[i].first << ": " << similarities[i].second * 100.0 << "%" << endl;
        }
        cout << "------------------------" << endl;
    }
}

int main()
{
    // Hyperparameters
    const int NUM_EPOCHS = 100;
    const double INITIAL_LEARNING_RATE = 0.05;
    const int WINDOW_SIZE = 2;
    const int NEG_SAMPLES = 5;
    const double SUBSAMPLE_THRESHOLD = 1e-5;
    const int EMBEDDING_DIM = 100;

    // Read corpus from a text file
    vector<string> words;
    ifstream file("output.txt");
    if (file.is_open())
    {
        string word;
        while (file >> word)
            words.push_back(word);
        file.close();
    }
    else
    {
        cerr << "Error: Unable to open file 'output.txt'" << endl;
        return 1;
    }

    cout << "Corpus size: " << words.size() << " words" << endl;
    
    // Build vocabulary (unique words)
    vector<string> vocabulary = buildVocabulary(words);
    cout << "Vocabulary size: " << vocabulary.size() << " unique words" << endl;

    // Build word-to-index map
    unordered_map<string, size_t> wordToIndex;
    for (size_t i = 0; i < vocabulary.size(); ++i)
        wordToIndex[vocabulary[i]] = i;
        
    // Compute word frequencies for subsampling
    unordered_map<string, double> wordFrequencies = computeWordFrequencies(words);

    // Initialize embedding and context matrices
    cout << "Initializing matrices with embedding dimension: " << EMBEDDING_DIM << endl;
    
    // Use Xavier/Glorot initialization for better training
    double xavier_limit = sqrt(6.0 / (vocabulary.size() + EMBEDDING_DIM));
    
    vector<vector<double>> embeddingMatrix(vocabulary.size(), vector<double>(EMBEDDING_DIM));
    vector<vector<double>> contextMatrix(EMBEDDING_DIM, vector<double>(vocabulary.size()));
    
    // Initialize with Xavier/Glorot initialization
    mt19937 rng(time(nullptr));
    uniform_real_distribution<double> uniform(-xavier_limit, xavier_limit);
    
    for (size_t i = 0; i < vocabulary.size(); i++) {
        for (size_t j = 0; j < EMBEDDING_DIM; j++) {
            embeddingMatrix[i][j] = uniform(rng);
        }
    }
    
    for (size_t i = 0; i < EMBEDDING_DIM; i++) {
        for (size_t j = 0; j < vocabulary.size(); j++) {
            contextMatrix[i][j] = uniform(rng);
        }
    }
    
    // Save initial matrices (optional)
    ofstream embOut("embedding_matrix_initial.txt");
    for (const auto &row : embeddingMatrix) {
        for (size_t j = 0; j < row.size(); j++) {
            embOut << row[j] << (j + 1 < row.size() ? " " : "");
        }
        embOut << endl;
    }
    embOut.close();
    
    ofstream ctxOut("context_matrix_initial.txt");
    for (const auto &row : contextMatrix) {
        for (size_t j = 0; j < row.size(); j++) {
            ctxOut << row[j] << (j + 1 < row.size() ? " " : "");
        }
        ctxOut << endl;
    }
    ctxOut.close();

    cout << "Training Word2Vec model with:" << endl;
    cout << "- Window size: " << WINDOW_SIZE << endl;
    cout << "- Negative samples: " << NEG_SAMPLES << endl;
    cout << "- Subsampling threshold: " << SUBSAMPLE_THRESHOLD << endl;
    cout << "- Initial learning rate: " << INITIAL_LEARNING_RATE << endl;
    
    // Train model with all features
    trainModel(embeddingMatrix, contextMatrix, words, vocabulary, wordToIndex, 
               wordFrequencies, NUM_EPOCHS, INITIAL_LEARNING_RATE, 
               WINDOW_SIZE, NEG_SAMPLES, SUBSAMPLE_THRESHOLD);

    // Save final matrices
    ofstream embOutFinal("embedding_matrix_final.txt");
    for (const auto &row : embeddingMatrix) {
        for (size_t j = 0; j < row.size(); j++) {
            embOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
        }
        embOutFinal << endl;
    }
    embOutFinal.close();
    
    ofstream ctxOutFinal("context_matrix_final.txt");
    for (const auto &row : contextMatrix) {
        for (size_t j = 0; j < row.size(); j++) {
            ctxOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
        }
        ctxOutFinal << endl;
    }
    ctxOutFinal.close();
    
    cout << "Training complete. Embedding and context matrices saved." << endl;

    // Interactive prediction
    interactivePrediction(embeddingMatrix, contextMatrix, vocabulary, wordToIndex);

    return 0;
}