#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <limits>

using namespace std;

// Function to compute cosine similarity between two vectors
double cosineSimilarity(const vector<double> &vec1, const vector<double> &vec2)
{
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (size_t i = 0; i < vec1.size(); i++)
    {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    // Prevent division by zero
    if (norm1 == 0.0 || norm2 == 0.0)
        return 0.0;

    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}

// Function to find the most similar word from the context matrix
string findMostSimilarWord(const vector<double> &wordEmbedding,
                           const vector<vector<double>> &contextMatrix,
                           const vector<string> &vocabulary)
{
    double maxSimilarity = -1.0;
    size_t bestWordIndex = 0;

    // For each word in vocabulary, compute similarity with input word
    for (size_t i = 0; i < vocabulary.size(); i++)
    {
        // Get the context vector for this word
        vector<double> contextVector(contextMatrix.size());
        for (size_t j = 0; j < contextMatrix.size(); j++)
        {
            contextVector[j] = contextMatrix[j][i];
        }

        // Compute cosine similarity
        double similarity = cosineSimilarity(wordEmbedding, contextVector);

        // Update best match if this is more similar
        if (similarity > maxSimilarity)
        {
            maxSimilarity = similarity;
            bestWordIndex = i;
        }
    }

    return vocabulary[bestWordIndex];
}

void fillMatrixWithRandomValues(vector<vector<double>> &matrix, int rows, int cols)
{
    srand(static_cast<unsigned>(time(0))); // Seed for random number generation
    matrix.resize(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1
        }
    }
}

int main()
{
    // Hyperparameters
    const int NUM_EPOCHS = 100;
    const double LEARNING_RATE = 0.1;

    // Expanded corpus
    vector<vector<string>> words = {
        {"dog", "bites", "man"},
        {"man", "bites", "lollipop"},
        {"dog", "eats", "meat"},
        {"man", "eats", "food"},
        {"cat", "chases", "mouse"},
        {"mouse", "eats", "cheese"},
        {"bird", "flies", "high"},
        {"fish", "swims", "fast"},
        {"child", "plays", "game"},
        {"teacher", "teaches", "student"},
        {"student", "learns", "lesson"},
        {"car", "drives", "fast"},
        {"train", "travels", "far"},
        {"sun", "shines", "bright"},
        {"moon", "glows", "softly"},
        {"rain", "falls", "gently"},
        {"wind", "blows", "strong"},
        {"fire", "burns", "hot"},
        {"water", "flows", "smoothly"},
        {"earth", "rotates", "slowly"}};

    // Create a vocabulary from the corpus
    vector<string> vocabulary;
    for (const auto &sentence : words)
    {
        for (const auto &word : sentence)
        {
            if (find(vocabulary.begin(), vocabulary.end(), word) == vocabulary.end())
            {
                vocabulary.push_back(word); // Add unique words to the vocabulary
            }
        }
    }

    // Print the vocabulary
    cout << "Vocabulary:" << endl;
    for (const auto &word : vocabulary)
    {
        cout << word << endl;
    }
    cout << "------------------------" << endl;

    // embedding matrix
    vector<vector<double>> embeddingMatrix;
    fillMatrixWithRandomValues(embeddingMatrix, vocabulary.size(), 3); // Words in vocabulary, 3 dimensions

    // context matrix
    vector<vector<double>> contextMatrix;
    fillMatrixWithRandomValues(contextMatrix, 3, vocabulary.size()); // 3 dimensions, words in vocabulary

    cout << "Embedding Matrix:" << endl;
    for (size_t i = 0; i < embeddingMatrix.size(); i++)
    {
        cout << vocabulary[i] << ": ";
        for (const auto &val : embeddingMatrix[i])
        {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "------------------------" << endl;

    // Print the context matrix
    cout << "Context Matrix:" << endl;
    for (const auto &row : contextMatrix)
    {
        for (const auto &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "------------------------" << endl;

    // Training loop over multiple epochs
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        double totalLoss = 0.0;

        // Training for each sentence
        for (size_t i = 0; i < words.size(); ++i)
        {
            for (size_t j = 0; j < words[i].size(); ++j)
            {
                // One-hot encoding for the current word
                vector<double> oneHot(vocabulary.size(), 0.0);
                auto it = find(vocabulary.begin(), vocabulary.end(), words[i][j]);
                if (it == vocabulary.end())
                    continue; // Skip unknown words

                size_t currentWordIndex = distance(vocabulary.begin(), it);
                oneHot[currentWordIndex] = 1.0;

                // FORWARD PASS

                // Multiply one-hot encoding with embedding matrix to get embedding vector
                vector<double> embeddingResult(embeddingMatrix[0].size(), 0.0);
                for (size_t k = 0; k < embeddingMatrix[0].size(); ++k)
                {
                    for (size_t l = 0; l < vocabulary.size(); ++l)
                    {
                        embeddingResult[k] += oneHot[l] * embeddingMatrix[l][k];
                    }
                }

                // Multiply the embedding vector with context matrix to get scores
                vector<double> contextResult(vocabulary.size(), 0.0);
                for (size_t k = 0; k < vocabulary.size(); ++k)
                {
                    for (size_t l = 0; l < embeddingResult.size(); ++l)
                    {
                        contextResult[k] += embeddingResult[l] * contextMatrix[l][k];
                    }
                }

                // Apply softmax to get probabilities
                vector<double> softmaxResult(vocabulary.size(), 0.0);
                double sumExp = 0.0;
                for (const auto &val : contextResult)
                {
                    sumExp += exp(val);
                }
                for (size_t k = 0; k < contextResult.size(); ++k)
                {
                    softmaxResult[k] = exp(contextResult[k]) / sumExp;
                }

                // Determine the target word (next word in the sentence)
                if (j + 1 < words[i].size())
                {
                    string nextWord = words[i][j + 1];
                    auto targetIt = find(vocabulary.begin(), vocabulary.end(), nextWord);
                    if (targetIt == vocabulary.end())
                        continue; // Skip unknown words

                    size_t targetWordIndex = distance(vocabulary.begin(), targetIt);

                    // Compute cross-entropy loss
                    double crossEntropyLoss = -log(softmaxResult[targetWordIndex]);
                    totalLoss += crossEntropyLoss;

                    if (epoch == NUM_EPOCHS - 1) // Only print in last epoch
                    {
                        cout << words[i][j] << " -> " << nextWord << ": Loss = " << crossEntropyLoss << endl;
                    }

                    // BACKPROPAGATION STEP

                    // 1. Compute error gradient at softmax output
                    // Create a copy of softmax output for computing gradient
                    vector<double> softmaxGradient = softmaxResult;
                    // For true class, subtract 1 (derivative of cross-entropy with softmax)
                    softmaxGradient[targetWordIndex] -= 1.0;

                    // 2. Gradients for the context matrix
                    // For each dimension in the embedding vector and each word in vocabulary
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                    {
                        for (size_t w = 0; w < vocabulary.size(); w++)
                        {
                            // Update context matrix using gradient descent
                            contextMatrix[d][w] -= LEARNING_RATE * embeddingResult[d] * softmaxGradient[w];
                        }
                    }

                    // 3. Compute gradient for embedding vector
                    vector<double> embeddingGradient(embeddingResult.size(), 0.0);
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                    {
                        for (size_t w = 0; w < vocabulary.size(); w++)
                        {
                            embeddingGradient[d] += softmaxGradient[w] * contextMatrix[d][w];
                        }
                    }

                    // 4. Update embedding matrix (only for the current word)
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                    {
                        embeddingMatrix[currentWordIndex][d] -= LEARNING_RATE * embeddingGradient[d];
                    }
                }
            }
        }

        // Print epoch progress every 10 epochs
        if (epoch % 10 == 0 || epoch == NUM_EPOCHS - 1)
        {
            cout << "Epoch " << epoch << ", Average Loss: " << totalLoss / (words.size() * 2) << endl;
        }
    }

    cout << "------------------------" << endl;
    // Print the final embedding matrix
    cout << "Embedding Matrix:" << endl;
    for (size_t i = 0; i < embeddingMatrix.size(); i++)
    {
        cout << vocabulary[i] << ": ";
        for (const auto &val : embeddingMatrix[i])
        {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "------------------------" << endl;

    // Print the final context matrix
    cout << "Context Matrix:" << endl;
    for (const auto &row : contextMatrix)
    {
        for (const auto &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
    cout << "------------------------" << endl;

    // Interactive word prediction loop
    cout << "Word Prediction Mode (enter 'exit' to quit)" << endl;
    cout << "------------------------" << endl;

    string inputWord;
    while (true)
    {
        cout << "Enter a word: ";
        cin >> inputWord;

        if (inputWord == "exit")
            break;

        // Check if word is in vocabulary
        auto wordIt = find(vocabulary.begin(), vocabulary.end(), inputWord);
        if (wordIt == vocabulary.end())
        {
            cout << "Word not in vocabulary. Please try another word." << endl;
            continue;
        }

        // Get the word embedding
        size_t wordIndex = distance(vocabulary.begin(), wordIt);
        vector<double> wordEmbedding = embeddingMatrix[wordIndex];

        // Find most similar word based on cosine similarity
        string predictedWord = findMostSimilarWord(wordEmbedding, contextMatrix, vocabulary);

        // Compute cosine similarity for the predicted word
        auto predictedWordIt = find(vocabulary.begin(), vocabulary.end(), predictedWord);
        size_t predictedWordIndex = distance(vocabulary.begin(), predictedWordIt);
        vector<double> predictedWordEmbedding = embeddingMatrix[predictedWordIndex];
        double similarity = cosineSimilarity(wordEmbedding, predictedWordEmbedding);

        cout << "Next word prediction: " << predictedWord << " (" << similarity * 100 << "% similarity)" << endl;
    }

    return 0;
}