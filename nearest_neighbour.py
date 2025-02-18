import numpy as np
from typing import Optional, Tuple, List

from utils import calculate_balanced_accuracy


class NearestNeighbourClassifier:

    k: int
    training_data: dict
    training_confusion_matrix: List[List]

    def __init__(self, training_data: dict, k: Optional[int] = None):
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
            class_names = list(normalised_data.keys())
            for class_name in class_names:
                data = normalised_data[class_name]
                training_set[class_name] = data[:len(data)//2]
                validation_set[class_name] = data[len(data)//2:]

            print('Produced training/validation split')

            self.training_data = training_set
            best_k = -1
            best_balanced_accuracy = -1
            best_confusion_matrix = None
            num_data = sum([len(x) for x in [y for y in validation_set.values()]])
            for k in range(1, min(11, num_data), 2):
                self.k = k 
                confusion_matrix = np.zeros((len(class_names), len(class_names))).tolist()
                for class_name, list_of_image_features in validation_set.items():
                    for image_features in list_of_image_features:
                        predicted, _ = self.classify(image_features)
                        confusion_matrix[class_names.index(predicted.lower())][class_names.index(class_name.lower())] += 1

                current_balanced_accuracy = calculate_balanced_accuracy(confusion_matrix)
                if best_balanced_accuracy < current_balanced_accuracy:
                    best_balanced_accuracy = current_balanced_accuracy
                    best_k = k
                    best_confusion_matrix = confusion_matrix

                print(f'k with value {self.k} returned accuracy of {current_balanced_accuracy}')

            self.k = best_k
            self.training_confusion_matrix = best_confusion_matrix
            print(f"k-NN tuned with k-value {self.k} and training accuracy {best_balanced_accuracy}")
        else:
            self.k = k
        
        self.training_data = normalised_data

    def _do_classification(self, input):
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

        return predictions_count

    def classify(self, input) -> Tuple[str, float]:
        predictions_count = self._do_classification(input)

        prediction = max(predictions_count, key=predictions_count.get)
        confidence = predictions_count[prediction] / self.k

        return max(predictions_count, key=predictions_count.get), confidence

    def classify_top_x(self, input, x):
        predictions_count = self._do_classification(input)
        sorted_predictions = {k: v for k, v in sorted(predictions_count.items(), key=lambda item: item[1])}
        prediction_confidences = []
        i = 0
        for prediction, count in sorted_predictions.items():
            if i == x:
                break
            prediction_confidences.append((prediction, count/self.k))
            i += 1
        return prediction_confidences
    
    
