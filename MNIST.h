#pragma once

#include "Random.h"

#include <fstream>
#include <algorithm>
#include <string>
#include <tuple>

float** read_mnist_images(std::string full_path, int number_of_images) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };



    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        int image_size = n_rows * n_cols;

        unsigned char* buffer = new unsigned char[image_size];
        float** dataset = new float* [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            file.read((char*)buffer, image_size);
            dataset[i] = new float[image_size];
            for (int j = 0; j < image_size; j++) {
                dataset[i][j] = static_cast<float>(buffer[j]) / 127.5f - 1.0f;
            }
        }

        delete[] buffer;

        return dataset;
    }
    else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float** read_mnist_labels(std::string full_path, int number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        float** dataset = new float* [number_of_labels];
        char* buffer = new char[number_of_labels];
        file.read(buffer, number_of_labels);
        for (int i = 0; i < number_of_labels; i++) {
            dataset[i] = new float[10];
            std::fill(dataset[i], dataset[i] + 10, -1.0f);
            dataset[i][buffer[i]] = 1.0f;
        }

        delete[] buffer;
        return dataset;
    }
    else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

std::tuple<float**, float**> create_batches(float** datapoints, float** labels, int nItems, int batchSize)
{
    int nBatches = nItems / batchSize;
    float** batchedPoints = new float* [nBatches];
    float** batchedLabels = new float* [nBatches];

    std::vector<int> newIDs;
    newIDs.reserve(nItems);
    newIDs.resize(nItems);
    for (int i = 0; i < nItems; i++) newIDs[i] = i;
    std::shuffle(newIDs.begin(), newIDs.end(), generator);

    for (int i = 0; i < nBatches; i++) {

        batchedPoints[i] = new float[784 * batchSize];
        batchedLabels[i] = new float[10 * batchSize];

        for (int j = 0; j < batchSize; j++) {
            int cID = i * batchSize + j;

            std::copy(datapoints[newIDs[cID]], datapoints[newIDs[cID]] + 784, batchedPoints[i] + j * 784);
            std::copy(labels[newIDs[cID]], labels[newIDs[cID]] + 10, batchedLabels[i] + j * 10);
        }
    }
    return { batchedPoints, batchedLabels };
}