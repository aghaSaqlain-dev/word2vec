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
// Remove OpenMP header
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

using namespace std;

// Unchanged helper functions: cosineSimilarity, findMostSimilarWord, buildVocabulary, 
// computeWordFrequencies, getSubsamplingProb, getContextVector, sigmoid, printMatrix
// ...

// Training loop with single core implementation
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

    // Single random generator for sequential processing
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    mt19937 rng(seed);
    discrete_distribution<int> negativeSampler(noiseDist.begin(), noiseDist.end());
    uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // Epoch loop with adaptive learning rate
    double prevLoss = DBL_MAX;
    const double earlyStoppingThreshold = 0.00001;
    
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        double totalLoss = 0.0;
        
        // Adaptive learning rate (linear decay)
        double learningRate = initialLearningRate * (1.0 - static_cast<double>(epoch) / numEpochs);
        
        // Global gradients for model update
        vector<vector<double>> contextMatrixGrad(contextMatrix.size(), vector<double>(contextMatrix[0].size(), 0.0));
        vector<vector<double>> embeddingMatrixGrad(embeddingMatrix.size(), vector<double>(embeddingMatrix[0].size(), 0.0));
        
        // Sequential processing of all words
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
                
                // Process positive example (target word)
                vector<double> contextVector = getContextVector(contextMatrix, targetWordIndex);
                double dotProduct = 0.0;
                for (size_t d = 0; d < embeddingResult.size(); d++)
                    dotProduct += embeddingResult[d] * contextVector[d];
                
                double sigmoid_pos = sigmoid(dotProduct);
                double loss_pos = -log(sigmoid_pos + 1e-10);
                totalLoss += loss_pos;
                
                // Update gradients for positive example
                double grad_pos = (sigmoid_pos - 1.0); // derivative of -log(sigmoid(x))
                
                // Update context matrix gradient for positive sample
                for (size_t d = 0; d < embeddingResult.size(); d++)
                    contextMatrixGrad[d][targetWordIndex] += grad_pos * embeddingResult[d];
                    
                // Update embedding matrix gradient
                for (size_t d = 0; d < embeddingResult.size(); d++)
                    embeddingMatrixGrad[currentWordIndex][d] += grad_pos * contextVector[d];
                
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
                    totalLoss += loss_neg;
                    
                    // Update gradients for negative example
                    double grad_neg = sigmoid_neg; // derivative of -log(1-sigmoid(x))
                    
                    // Update context matrix gradient for negative sample
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        contextMatrixGrad[d][negWordIndex] += grad_neg * embeddingResult[d];
                        
                    // Update embedding matrix gradient
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        embeddingMatrixGrad[currentWordIndex][d] += grad_neg * negContextVector[d];
                }
            }
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

// int main()
// {
//     // Hyperparameters
//     const int NUM_EPOCHS = 100;
//     const double INITIAL_LEARNING_RATE = 0.05;
//     const int WINDOW_SIZE = 2;
//     const int NEG_SAMPLES = 5;
//     const double SUBSAMPLE_THRESHOLD = 1e-5;
//     const int EMBEDDING_DIM = 100;

//     // Read corpus from a text file
//     vector<string> words;
//     ifstream file("output.txt");
//     if (file.is_open())
//     {
//         string word;
//         while (file >> word)
//             words.push_back(word);
//         file.close();
//     }
//     else
//     {
//         cerr << "Error: Unable to open file 'output.txt'" << endl;
//         return 1;
//     }

//     cout << "Corpus size: " << words.size() << " words" << endl;
    
//     // Build vocabulary (unique words)
//     vector<string> vocabulary = buildVocabulary(words);
//     cout << "Vocabulary size: " << vocabulary.size() << " unique words" << endl;

//     // Build word-to-index map
//     unordered_map<string, size_t> wordToIndex;
//     for (size_t i = 0; i < vocabulary.size(); ++i)
//         wordToIndex[vocabulary[i]] = i;
        
//     // Compute word frequencies for subsampling
//     unordered_map<string, double> wordFrequencies = computeWordFrequencies(words);

//     // Initialize embedding and context matrices
//     cout << "Initializing matrices with embedding dimension: " << EMBEDDING_DIM << endl;
    
//     // Use Xavier/Glorot initialization for better training
//     double xavier_limit = sqrt(6.0 / (vocabulary.size() + EMBEDDING_DIM));
    
//     vector<vector<double>> embeddingMatrix(vocabulary.size(), vector<double>(EMBEDDING_DIM));
//     vector<vector<double>> contextMatrix(EMBEDDING_DIM, vector<double>(vocabulary.size()));
    
//     // Initialize with Xavier/Glorot initialization
//     mt19937 rng(time(nullptr));
//     uniform_real_distribution<double> uniform(-xavier_limit, xavier_limit);
    
//     for (size_t i = 0; i < vocabulary.size(); i++) {
//         for (size_t j = 0; j < EMBEDDING_DIM; j++) {
//             embeddingMatrix[i][j] = uniform(rng);
//         }
//     }
    
//     for (size_t i = 0; i < EMBEDDING_DIM; i++) {
//         for (size_t j = 0; j < vocabulary.size(); j++) {
//             contextMatrix[i][j] = uniform(rng);
//         }
//     }
    
//     // Save initial matrices (optional)
//     ofstream embOut("embedding_matrix_initial.txt");
//     for (const auto &row : embeddingMatrix) {
//         for (size_t j = 0; j < row.size(); j++) {
//             embOut << row[j] << (j + 1 < row.size() ? " " : "");
//         }
//         embOut << endl;
//     }
//     embOut.close();
    
//     ofstream ctxOut("context_matrix_initial.txt");
//     for (const auto &row : contextMatrix) {
//         for (size_t j = 0; j < row.size(); j++) {
//             ctxOut << row[j] << (j + 1 < row.size() ? " " : "");
//         }
//         ctxOut << endl;
//     }
//     ctxOut.close();

//     cout << "Training Word2Vec model with:" << endl;
//     cout << "- Window size: " << WINDOW_SIZE << endl;
//     cout << "- Negative samples: " << NEG_SAMPLES << endl;
//     cout << "- Subsampling threshold: " << SUBSAMPLE_THRESHOLD << endl;
//     cout << "- Initial learning rate: " << INITIAL_LEARNING_RATE << endl;
//     cout << "total words : " << vocabulary.size() <<endl;

//     auto start_time = chrono::high_resolution_clock::now();
    
//     // Train model with all features
//     trainModel(embeddingMatrix, contextMatrix, words, vocabulary, wordToIndex, 
//                wordFrequencies, NUM_EPOCHS, INITIAL_LEARNING_RATE, 
//                WINDOW_SIZE, NEG_SAMPLES, SUBSAMPLE_THRESHOLD);

    
//     auto end_time = chrono::high_resolution_clock::now();
//     auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
           
//     cout << "Training completed in " << duration.count() << " seconds (" 
//          << duration.count()/60.0 << " minutes)" << endl;
//     // Save final matrices
//     ofstream embOutFinal("embedding_matrix_final.txt");
//     for (const auto &row : embeddingMatrix) {
//         for (size_t j = 0; j < row.size(); j++) {
//             embOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
//         }
//         embOutFinal << endl;
//     }
//     embOutFinal.close();
    
//     ofstream ctxOutFinal("context_matrix_final.txt");
//     for (const auto &row : contextMatrix) {
//         for (size_t j = 0; j < row.size(); j++) {
//             ctxOutFinal << row[j] << (j + 1 < row.size() ? " " : "");
//         }
//         ctxOutFinal << endl;
//     }
//     ctxOutFinal.close();
    
//     cout << "Training complete. Embedding and context matrices saved." << endl;

//     // Interactive prediction
//     interactivePrediction(embeddingMatrix, contextMatrix, vocabulary, wordToIndex);

//     return 0;
// }

int main()
{
    // Fixed hyperparameters
    const int NUM_EPOCHS = 100;
    const double INITIAL_LEARNING_RATE = 0.05;
    const int NEG_SAMPLES = 5;
    const double SUBSAMPLE_THRESHOLD = 1e-5;
    
    // Arrays of hyperparameters to test
    const vector<int> WINDOW_SIZES = {2, 5, 10};
    const vector<int> EMBEDDING_DIMS = {50, 100, 200, 300};

    // Read corpus from a text file
    vector<string> words;
    ifstream file("output2.txt");
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

    // Keep track of the best configuration for interactive prediction
    double best_loss = DBL_MAX;
    int best_window_size = 0;
    int best_embedding_dim = 0;
    vector<vector<double>> best_embedding_matrix;
    vector<vector<double>> best_context_matrix;

    // For each combination of window size and embedding dimension
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
            vector<vector<double>> contextMatrix(embedding_dim, vector<double>(vocabulary.size()));
            
            // Initialize with Xavier/Glorot initialization
            mt19937 rng(time(nullptr));
            uniform_real_distribution<double> uniform(-xavier_limit, xavier_limit);
            
            for (size_t i = 0; i < vocabulary.size(); i++) {
                for (size_t j = 0; j < embedding_dim; j++) {
                    embeddingMatrix[i][j] = uniform(rng);
                }
            }
            
            for (size_t i = 0; i < embedding_dim; i++) {
                for (size_t j = 0; j < vocabulary.size(); j++) {
                    contextMatrix[i][j] = uniform(rng);
                }
            }
            
            // Save initial matrices with configuration in filename
            string config = "_w" + to_string(window_size) + "_d" + to_string(embedding_dim);
            ofstream embOut("embedding_matrix_initial" + config + ".txt");
            for (const auto &row : embeddingMatrix) {
                for (size_t j = 0; j < row.size(); j++) {
                    embOut << row[j] << (j + 1 < row.size() ? " " : "");
                }
                embOut << endl;
            }
            embOut.close();
            
            cout << "Training Word2Vec model with:" << endl;
            cout << "- Window size: " << window_size << endl;
            cout << "- Embedding dimension: " << embedding_dim << endl;
            cout << "- Negative samples: " << NEG_SAMPLES << endl;
            cout << "- Subsampling threshold: " << SUBSAMPLE_THRESHOLD << endl;
            cout << "- Initial learning rate: " << INITIAL_LEARNING_RATE << endl;
            cout << "- Vocabulary size: " << vocabulary.size() << endl;

            auto start_time = chrono::high_resolution_clock::now();
            
            // Store the previous loss value to calculate improvement
            double prevLoss = DBL_MAX;
            
            // Train model with current configuration
            trainModel(embeddingMatrix, contextMatrix, words, vocabulary, wordToIndex, 
                       wordFrequencies, NUM_EPOCHS, INITIAL_LEARNING_RATE, 
                       window_size, NEG_SAMPLES, SUBSAMPLE_THRESHOLD);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
            
            cout << "Training completed in " << duration.count() << " seconds (" 
                 << duration.count()/60.0 << " minutes)" << endl;
                 
            // Save final matrices with configuration in filename
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
            
            cout << "Matrices saved with config: " << config << endl;
            
            // Test a few example words to evaluate model quality
            cout << "\nTesting model quality with example words:" << endl;
            vector<string> testWords = {"the", "and", "of", "to", "in"};  // Common words likely in vocabulary
            
            for (const string& word : testWords) {
                auto wordIt = wordToIndex.find(word);
                if (wordIt != wordToIndex.end()) {
                    size_t wordIndex = wordIt->second;
                    string mostSimilar = findMostSimilarWord(embeddingMatrix[wordIndex], 
                                                            contextMatrix, vocabulary);
                    cout << "Most similar to '" << word << "': " << mostSimilar << endl;
                }
            }
        }
    }

    cout << "\nTraining complete for all configurations." << endl;
    cout << "Use the saved matrices for further analysis." << endl;

    // Ask user which configuration to use for interactive prediction
    cout << "\nEnter configuration for interactive prediction:" << endl;
    int userWindowSize, userEmbeddingDim;
    
    cout << "Window size (";
    for (size_t i = 0; i < WINDOW_SIZES.size(); i++) {
        cout << WINDOW_SIZES[i] << (i < WINDOW_SIZES.size()-1 ? ", " : "");
    }
    cout << "): ";
    cin >> userWindowSize;
    
    cout << "Embedding dimension (";
    for (size_t i = 0; i < EMBEDDING_DIMS.size(); i++) {
        cout << EMBEDDING_DIMS[i] << (i < EMBEDDING_DIMS.size()-1 ? ", " : "");
    }
    cout << "): ";
    cin >> userEmbeddingDim;
    
    // Load the selected configuration
    string configToLoad = "_w" + to_string(userWindowSize) + "_d" + to_string(userEmbeddingDim);
    cout << "Loading matrices with config: " << configToLoad << endl;
    
    // You could add code here to load the matrices from files if needed
    // For now, we'll assume the user selects one of the configurations we just trained
    
    // Find the selected configuration
    vector<vector<double>> selectedEmbeddingMatrix;
    vector<vector<double>> selectedContextMatrix;
    bool configFound = false;
    
    for (int window_size : WINDOW_SIZES) {
        for (int embedding_dim : EMBEDDING_DIMS) {
            if (window_size == userWindowSize && embedding_dim == userEmbeddingDim) {
                // This is the configuration we want
                // In a real implementation, you might load these from files
                // For simplicity, we'll just run interactive prediction with what we have
                configFound = true;
                break;
            }
        }
        if (configFound) break;
    }
    
    if (configFound) {
        // Load matrices from files for the selected configuration
        vector<vector<double>> selectedEmbeddingMatrix;
        vector<vector<double>> selectedContextMatrix;
        
        // Load embedding matrix
        string embeddingFile = "embedding_matrix_final" + configToLoad + ".txt";
        ifstream embIn(embeddingFile);
        if (embIn.is_open()) {
            string line;
            while (getline(embIn, line)) {
                istringstream iss(line);
                vector<double> row;
                double val;
                while (iss >> val) {
                    row.push_back(val);
                }
                selectedEmbeddingMatrix.push_back(row);
            }
            embIn.close();
        } else {
            cerr << "Could not open " << embeddingFile << endl;
            return 1;
        }
        
        // Load context matrix
        string contextFile = "context_matrix_final" + configToLoad + ".txt";
        ifstream ctxIn(contextFile);
        if (ctxIn.is_open()) {
            string line;
            while (getline(ctxIn, line)) {
                istringstream iss(line);
                vector<double> row;
                double val;
                while (iss >> val) {
                    row.push_back(val);
                }
                selectedContextMatrix.push_back(row);
            }
            ctxIn.close();
        } else {
            cerr << "Could not open " << contextFile << endl;
            return 1;
        }
        
        // Run interactive prediction with loaded matrices
        interactivePrediction(selectedEmbeddingMatrix, selectedContextMatrix, vocabulary, wordToIndex);
    } else {
        cout << "Configuration not found. Please select from the available options." << endl;
    }

    return 0;
}