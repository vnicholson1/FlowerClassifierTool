import numpy as np
from typing import Optional, List


class NearestNeighbourClassifier:

    k: int
    training_data: dict

    def __init__(self, training_data: dict, k: Optional[int] = None):
        self.k = k
        # normalise the data before training
        normalised_data = {}
        for class_name, list_of_image_features in training_data.items(): 
            normalised_data[class_name] = []
            for image_features in list_of_image_features:
                normalised_data[class_name].append((image_features-np.min(image_features))/(np.max(image_features)-np.min(image_features)))

        print('Normalised the data')

        # Tune k with 50/50 CV split
        if k is None:
            training_set = {}
            validation_set = {}
            for class_name in normalised_data.keys():
                data = normalised_data[class_name]
                training_set[class_name] = data[:len(data)//2]
                validation_set[class_name] = data[len(data)//2:]

            print('Produced training/validation split')

            self.training_data = training_set
            best_k = -1
            best_correct = -1
            num_data = sum([len(x) for x in [y for y in validation_set.values()]])
            for k in range(5, min(50, num_data), 5):
                self.k = k 
                num_correct = 0
                for class_name, list_of_image_features in validation_set.items():
                    print(f'Currently predicting {class_name} images')
                    for image_features in list_of_image_features:
                        predicted = self.classify(image_features)
                        if predicted == class_name:
                            num_correct += 1
                if num_correct > best_correct:
                    best_correct = num_correct
                    best_k = k

                print(f'k with value {self.k} returned accuracy of {num_correct/num_data}')

            self.k = best_k
            print(f"k-NN tuned with k-value {self.k} and training accuracy {best_correct/num_data}")

        self.training_data = normalised_data

    def classify(self, input) -> str:
        normalised_input = (input-np.min(input))/(np.max(input)-np.min(input))
        best_distances = []
        best_classes = []
        for class_name, features in self.training_data.items():
            for data in features:
                distance = 0
                for x in range(len(data)):
                    distance += (data[x] - normalised_input[x]) ** 2
                    if len(best_distances) == self.k and distance > max(best_distances):
                        break

                if len(best_distances) < self.k:
                    best_distances.append(distance)
                    best_classes.append(class_name)
                elif len(best_distances) == self.k and distance < max(best_distances):
                    worst_k_distance = max(best_distances)
                    index_to_remove = best_distances.index(worst_k_distance)   

                    del best_distances[index_to_remove]
                    del best_classes[index_to_remove]

                    best_distances.append(distance)
                    best_classes.append(class_name)

        predictions_count = {}
        for class_name in best_classes:
            if class_name not in predictions_count:
                predictions_count[class_name] = 0
            predictions_count[class_name] += 1

        return max(predictions_count, key=predictions_count.get)
