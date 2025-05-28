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
#include <cfloat>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { printf("CUDA error: %s, %s, line %d\n", cudaGetErrorString(error), __FILE__, __LINE__); exit(1); } }

// Define batch size for processing
#define BATCH_SIZE 1024

// Forward declarations
double getSubsamplingProb(double freq, double threshold);
double sigmoid(double x);
vector<double> getContextVector(const vector<vector<double>> &contextMatrix, size_t wordIndex);

// CUDA kernel for sigmoid calculation
__global__ void sigmoidKernel(double* dot_products, double* sigmoid_outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        sigmoid_outputs[idx] = 1.0 / (1.0 + exp(-dot_products[idx]));
    }
}

// CUDA kernel for computing gradient updates
__global__ void gradientKernel(double* embeddings, double* contexts, double* gradients, 
                             int* word_indices, int* context_indices, double learning_rate,
                             int embedding_dim, int size, bool is_negative) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        int word_idx = word_indices[i];
        int ctx_idx = context_indices[i];
        double grad = gradients[i];
        
        // If this is a negative sample, we use different gradient
        if (is_negative) {
            grad = grad; // For negative samples, grad = sigmoid
        } else {
            grad = grad - 1.0; // For positive samples, grad = sigmoid - 1
        }
        
        for (int d = 0; d < embedding_dim; d++) {
            // Update both embedding and context vectors
            double embed_update = -learning_rate * grad * contexts[ctx_idx * embedding_dim + d];
            double context_update = -learning_rate * grad * embeddings[word_idx * embedding_dim + d];
            
            // Use atomic operations to avoid race conditions
            atomicAdd(&embeddings[word_idx * embedding_dim + d], embed_update);
            atomicAdd(&contexts[ctx_idx * embedding_dim + d], context_update);
        }
    }
}

// CUDA kernel for batch dot product calculation
__global__ void batchDotProductKernel(double* embeddings, double* contexts, 
                                    int* word_indices, int* context_indices,
                                    double* dot_products, int embedding_dim, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int word_idx = word_indices[idx];
        int ctx_idx = context_indices[idx];
        
        double dot = 0.0;
        for (int d = 0; d < embedding_dim; d++) {
            dot += embeddings[word_idx * embedding_dim + d] * contexts[ctx_idx * embedding_dim + d];
        }
        dot_products[idx] = dot;
    }
}

// Dot product kernel for shared memory reduction
__global__ void dotProductKernel(const double* a, const double* b, double* result, int size) {
    __shared__ double cache[256]; // Adjust size as needed
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(result, cache[0]);
}

// Replace lines 116-128 with this:
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

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
    int vocabSize = vocabulary.size();
    int embeddingDim = embeddingMatrix[0].size();
    
    // Flatten matrices for GPU (row-major layout)
    vector<double> flat_embeddings(vocabSize * embeddingDim);
    vector<double> flat_contexts(vocabSize * embeddingDim);
    
    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingDim; j++) {
            flat_embeddings[i * embeddingDim + j] = embeddingMatrix[i][j];
            flat_contexts[i * embeddingDim + j] = contextMatrix[i][j];
        }
    }
    
    // Allocate device memory ONCE (not in the loop)
    double *d_embeddings, *d_contexts;
    double *d_dot_products, *d_sigmoids;
    int *d_word_indices, *d_context_indices;
    
    // Allocate memory for matrices
    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabSize * embeddingDim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_contexts, vocabSize * embeddingDim * sizeof(double)));
    
    // Allocate memory for batch processing
    CUDA_CHECK(cudaMalloc(&d_dot_products, BATCH_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigmoids, BATCH_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_word_indices, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_context_indices, BATCH_SIZE * sizeof(int)));
    
    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_embeddings, flat_embeddings.data(), 
                        vocabSize * embeddingDim * sizeof(double), 
                        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_contexts, flat_contexts.data(), 
                        vocabSize * embeddingDim * sizeof(double), 
                        cudaMemcpyHostToDevice));
    
    // Prepare for negative sampling
    vector<double> noiseDist(vocabSize);
    double normFactor = 0.0;
    for (size_t i = 0; i < vocabSize; i++) {
        double freq = wordFrequencies.at(vocabulary[i]);
        noiseDist[i] = pow(freq, 0.75);
        normFactor += noiseDist[i];
    }
    for (double &val : noiseDist) val /= normFactor;
    
    mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    discrete_distribution<int> negSampler(noiseDist.begin(), noiseDist.end());
    uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // Host arrays for batch processing
    vector<int> batch_word_indices(BATCH_SIZE);
    vector<int> batch_context_indices(BATCH_SIZE);
    vector<double> batch_dot_products(BATCH_SIZE);
    vector<double> batch_sigmoids(BATCH_SIZE);
    
    // Training loop
    double prevLoss = DBL_MAX;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        double learningRate = initialLearningRate * (1.0 - static_cast<double>(epoch) / numEpochs);
        double totalLoss = 0.0;
        int batch_count = 0;
        
        // Create training pairs for this epoch
        vector<pair<int, int>> positive_pairs;
        for (size_t i = 0; i < words.size(); ++i) {
            auto it = wordToIndex.find(words[i]);
            if (it == wordToIndex.end()) continue;
            int currentWordIdx = it->second;
            double freq = wordFrequencies.at(words[i]);
            if (uniform(rng) > getSubsamplingProb(freq, subsampleThreshold)) continue;
            
            for (int j = -windowSize; j <= windowSize; ++j) {
                if (j == 0) continue;
                size_t ctxPos = i + j;
                if (ctxPos >= words.size()) continue;
                
                auto tgtIt = wordToIndex.find(words[ctxPos]);
                if (tgtIt == wordToIndex.end()) continue;
                int targetIdx = tgtIt->second;
                
                positive_pairs.push_back({currentWordIdx, targetIdx});
            }
        }
        
        // Shuffle positive pairs for better training
        shuffle(positive_pairs.begin(), positive_pairs.end(), rng);
        
        // Process in batches
        for (size_t pair_idx = 0; pair_idx < positive_pairs.size(); pair_idx += BATCH_SIZE) {
            int current_batch_size = min(BATCH_SIZE, (int)(positive_pairs.size() - pair_idx));
            
            // Fill batch with positive samples
            for (int i = 0; i < current_batch_size; i++) {
                batch_word_indices[i] = positive_pairs[pair_idx + i].first;
                batch_context_indices[i] = positive_pairs[pair_idx + i].second;
            }
            
            // Copy batch to device
            CUDA_CHECK(cudaMemcpy(d_word_indices, batch_word_indices.data(), 
                               current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_context_indices, batch_context_indices.data(), 
                               current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            // Calculate dot products for positive samples
            batchDotProductKernel<<<(current_batch_size + 255)/256, 256>>>(
                d_embeddings, d_contexts, d_word_indices, d_context_indices,
                d_dot_products, embeddingDim, current_batch_size);
            
            // Calculate sigmoids
            sigmoidKernel<<<(current_batch_size + 255)/256, 256>>>(
                d_dot_products, d_sigmoids, current_batch_size);
            
            // Copy results back to host
            CUDA_CHECK(cudaMemcpy(batch_sigmoids.data(), d_sigmoids, 
                               current_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
            
            // Update gradients for positive samples
            gradientKernel<<<32, 256>>>(d_embeddings, d_contexts, d_sigmoids, 
                                      d_word_indices, d_context_indices, learningRate,
                                      embeddingDim, current_batch_size, false);
            
            // Calculate loss for positive samples
            for (int i = 0; i < current_batch_size; i++) {
                totalLoss += -log(batch_sigmoids[i] + 1e-10);
            }
            
            // Process negative samples
            for (int neg = 0; neg < negSamples; neg++) {
                // Generate negative samples
                for (int i = 0; i < current_batch_size; i++) {
                    batch_context_indices[i] = negSampler(rng);
                    
                    // Make sure negative sample is not the positive context
                    while (batch_context_indices[i] == positive_pairs[pair_idx + i].second) {
                        batch_context_indices[i] = negSampler(rng);
                    }
                }
                
                // Copy negative context indices to device
                CUDA_CHECK(cudaMemcpy(d_context_indices, batch_context_indices.data(), 
                                   current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
                
                // Calculate dot products for negative samples
                batchDotProductKernel<<<(current_batch_size + 255)/256, 256>>>(
                    d_embeddings, d_contexts, d_word_indices, d_context_indices,
                    d_dot_products, embeddingDim, current_batch_size);
                
                // Calculate sigmoids for negative samples
                sigmoidKernel<<<(current_batch_size + 255)/256, 256>>>(
                    d_dot_products, d_sigmoids, current_batch_size);
                
                // Copy results back to host
                CUDA_CHECK(cudaMemcpy(batch_sigmoids.data(), d_sigmoids, 
                                   current_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
                
                // Update gradients for negative samples
                gradientKernel<<<32, 256>>>(d_embeddings, d_contexts, d_sigmoids, 
                                          d_word_indices, d_context_indices, learningRate,
                                          embeddingDim, current_batch_size, true);
                
                // Calculate loss for negative samples
                for (int i = 0; i < current_batch_size; i++) {
                    totalLoss += -log(1.0 - batch_sigmoids[i] + 1e-10);
                }
            }
            
            batch_count++;
            if (batch_count % 100 == 0) {
                cout << "Epoch " << epoch << ", Batch " << batch_count 
                     << ", Processed " << batch_count * BATCH_SIZE << " examples" << endl;
            }
        }
        
        // Copy updated matrices back to host
        CUDA_CHECK(cudaMemcpy(flat_embeddings.data(), d_embeddings, 
                           vocabSize * embeddingDim * sizeof(double), 
                           cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flat_contexts.data(), d_contexts, 
                           vocabSize * embeddingDim * sizeof(double), 
                           cudaMemcpyDeviceToHost));
        
        // Convert flat arrays back to matrices
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddingMatrix[i][j] = flat_embeddings[i * embeddingDim + j];
                contextMatrix[i][j] = flat_contexts[i * embeddingDim + j];
            }
        }
        
        cout << "Epoch " << epoch << ", Learning Rate: " << learningRate
             << ", Avg Loss: " << totalLoss / words.size() << endl;
             
        // Early stopping
        if (epoch > 10 && abs(prevLoss - totalLoss) < 0.01 * prevLoss) {
            cout << "Early stopping at epoch " << epoch << endl;
            break;
        }
        prevLoss = totalLoss;
    }
    
    // Free device memory
    cudaFree(d_embeddings);
    cudaFree(d_contexts);
    cudaFree(d_dot_products);
    cudaFree(d_sigmoids);
    cudaFree(d_word_indices);
    cudaFree(d_context_indices);
}

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
        double similarity = cosineSimilarity(wordEmbedding, contextMatrix[i]);
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
double getSubsamplingProb(double freq, double threshold)
{
    if (freq < threshold)
        return 1.0;
    return (sqrt(freq / threshold) + 1) * threshold / freq;
}

// Helper to get a context vector from context matrix
vector<double> getContextVector(const vector<vector<double>> &contextMatrix, size_t wordIndex)
{
    return contextMatrix[wordIndex];
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
        const vector<double> &wordEmbedding = embeddingMatrix[wordIndex];

        vector<pair<string, double>> similarities;
        for (size_t i = 0; i < vocabulary.size(); i++)
        {
            double similarity = cosineSimilarity(wordEmbedding, contextMatrix[i]);
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
    // Arrays of hyperparameters to test
    const vector<int> WINDOW_SIZES = {2};
    const vector<int> EMBEDDING_DIMS = {50, 100};
    
    // Fixed hyperparameters
    const int NUM_EPOCHS = 100;
    const double INITIAL_LEARNING_RATE = 0.05;
    const int NEG_SAMPLES = 5;
    const double SUBSAMPLE_THRESHOLD = 1e-5;

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

    // Run all combinations
    for (int window_size : WINDOW_SIZES) {
        for (int embedding_dim : EMBEDDING_DIMS) {
            cout << "\n=======================================================" << endl;
            cout << "TRAINING WITH WINDOW SIZE: " << window_size 
                 << " AND EMBEDDING DIM: " << embedding_dim << endl;
            cout << "=======================================================" << endl;
            
            // Initialize embedding and context matrices
            cout << "Initializing matrices with embedding dimension: " << embedding_dim << endl;
            
            // Use Xavier/Glorot initialization for better training
            double xavier_limit = sqrt(6.0 / (vocabulary.size() + embedding_dim));
            
            vector<vector<double>> embeddingMatrix(vocabulary.size(), vector<double>(embedding_dim));
            vector<vector<double>> contextMatrix(vocabulary.size(), vector<double>(embedding_dim));
            
            // Initialize with Xavier/Glorot initialization
            mt19937 rng(time(nullptr));
            uniform_real_distribution<double> uniform(-xavier_limit, xavier_limit);
            
            for (size_t i = 0; i < vocabulary.size(); i++) {
                for (size_t j = 0; j < embedding_dim; j++) {
                    embeddingMatrix[i][j] = uniform(rng);
                    contextMatrix[i][j] = uniform(rng);
                }
            }
            
            // Create configuration string for file naming
            string config = "_w" + to_string(window_size) + "_d" + to_string(embedding_dim);
            
            // Save initial matrices
            ofstream embOut("embedding_matrix_initial" + config + ".txt");
            for (const auto &row : embeddingMatrix) {
                for (size_t j = 0; j < row.size(); j++) {
                    embOut << row[j] << (j + 1 < row.size() ? " " : "");
                }
                embOut << endl;
            }
            embOut.close();
            
            ofstream ctxOut("context_matrix_initial" + config + ".txt");
            for (const auto &row : contextMatrix) {
                for (size_t j = 0; j < row.size(); j++) {
                    ctxOut << row[j] << (j + 1 < row.size() ? " " : "");
                }
                ctxOut << endl;
            }
            ctxOut.close();

            cout << "Training Word2Vec model with:" << endl;
            cout << "- Window size: " << window_size << endl;
            cout << "- Embedding dimension: " << embedding_dim << endl;
            cout << "- Negative samples: " << NEG_SAMPLES << endl;
            cout << "- Subsampling threshold: " << SUBSAMPLE_THRESHOLD << endl;
            cout << "- Initial learning rate: " << INITIAL_LEARNING_RATE << endl;
            cout << "- Vocabulary size: " << vocabulary.size() << endl;

            auto start_time = chrono::high_resolution_clock::now();
            
            // Train model with all features
            trainModel(embeddingMatrix, contextMatrix, words, vocabulary, wordToIndex, 
                      wordFrequencies, NUM_EPOCHS, INITIAL_LEARNING_RATE, 
                      window_size, NEG_SAMPLES, SUBSAMPLE_THRESHOLD);

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

            cout << "Training completed in " << duration.count() << " seconds (" 
                 << duration.count()/60.0 << " minutes)" << endl;
                 
            // Save final matrices
            ofstream embOutFinal("embedding_matrix_final" + config + ".txt");
            for (const auto &row : embeddingMatrix) {
                for (size_t j = 0; j < row.size(); j++) {
                    embOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
                }
                embOutFinal << endl;
            }
            embOutFinal.close();
            
            ofstream ctxOutFinal("context_matrix_final" + config + ".txt");
            for (const auto &row : contextMatrix) {
                for (size_t j = 0; j < row.size(); j++) {
                    ctxOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
                }
                ctxOutFinal << endl;
            }
            ctxOutFinal.close();
            
            cout << "Matrices saved with configuration: " << config << endl;
            
            // Test a few example words
            cout << "\nTesting a few example words:" << endl;
            vector<string> testWords = {"the", "and", "of"};
            
            for (const string& word : testWords) {
                auto wordIt = wordToIndex.find(word);
                if (wordIt != wordToIndex.end()) {
                    size_t wordIndex = wordIt->second;
                    
                    vector<pair<string, double>> similarities;
                    for (size_t i = 0; i < vocabulary.size(); i++) {
                        double similarity = cosineSimilarity(embeddingMatrix[wordIndex], contextMatrix[i]);
                        similarities.emplace_back(vocabulary[i], similarity);
                    }
                    
                    sort(similarities.begin(), similarities.end(),
                         [](const pair<string, double> &a, const pair<string, double> &b) {
                             return a.second > b.second;
                         });
                    
                    cout << "Top 5 similar words to '" << word << "':" << endl;
                    for (size_t i = 0; i < min(size_t(5), similarities.size()); i++) {
                        cout << similarities[i].first << ": " << similarities[i].second * 100.0 << "%" << endl;
                    }
                    cout << "------------------------" << endl;
                }
            }
        }
    }

    // Ask user which configuration to use for interactive prediction
    cout << "\nWhich configuration would you like to use for interactive prediction?" << endl;
    
    int selectedWindowSize;
    int selectedEmbeddingDim;
    
    cout << "Enter window size (";
    for (size_t i = 0; i < WINDOW_SIZES.size(); i++) {
        cout << WINDOW_SIZES[i] << (i < WINDOW_SIZES.size()-1 ? ", " : "");
    }
    cout << "): ";
    cin >> selectedWindowSize;
    
    cout << "Enter embedding dimension (";
    for (size_t i = 0; i < EMBEDDING_DIMS.size(); i++) {
        cout << EMBEDDING_DIMS[i] << (i < EMBEDDING_DIMS.size()-1 ? ", " : "");
    }
    cout << "): ";
    cin >> selectedEmbeddingDim;
    
    // Load matrices for selected configuration
    string selectedConfig = "_w" + to_string(selectedWindowSize) + "_d" + to_string(selectedEmbeddingDim);
    cout << "Loading configuration: " << selectedConfig << endl;
    
    vector<vector<double>> selectedEmbeddingMatrix(vocabulary.size(), vector<double>(selectedEmbeddingDim));
    vector<vector<double>> selectedContextMatrix(vocabulary.size(), vector<double>(selectedEmbeddingDim));
    
    // Load embedding matrix
    ifstream embIn("embedding_matrix_final" + selectedConfig + ".txt");
    if (embIn.is_open()) {
        for (size_t i = 0; i < vocabulary.size(); i++) {
            for (size_t j = 0; j < selectedEmbeddingDim; j++) {
                embIn >> selectedEmbeddingMatrix[i][j];
            }
        }
        embIn.close();
    } else {
        cerr << "Error loading embedding matrix" << endl;
        return 1;
    }
    
    // Load context matrix
    ifstream ctxIn("context_matrix_final" + selectedConfig + ".txt");
    if (ctxIn.is_open()) {
        for (size_t i = 0; i < vocabulary.size(); i++) {
            for (size_t j = 0; j < selectedEmbeddingDim; j++) {
                ctxIn >> selectedContextMatrix[i][j];
            }
        }
        ctxIn.close();
    } else {
        cerr << "Error loading context matrix" << endl;
        return 1;
    }
    
    // Run interactive prediction with selected configuration
    interactivePrediction(selectedEmbeddingMatrix, selectedContextMatrix, vocabulary, wordToIndex);

    return 0;
}