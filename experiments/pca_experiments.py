import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_image(img_path, image_size):
    """Load an image and preprocess it into a flattened NumPy array."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(image_size)  # Resize to a fixed size
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize pixel values
    return img_array.flatten()

def extract_features_from_directory(directory, image_size):
    """Extract features from all images in a directory where subfolders represent class labels."""
    features_list = []
    labels = []
    class_names = sorted(os.listdir(directory))  # Ensure consistent label ordering
    
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_path, filename)
                    features = load_and_preprocess_image(img_path, image_size)
                    features_list.append(features)
                    labels.append(class_name)
    
    return np.array(features_list), np.array(labels)

def reduce_dimensionality(features, n_components=20):
    """Apply PCA to reduce feature dimensionality."""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

def tune_hyperparameters(X_train, y_train, classifier, param_grid):
    """Perform hyperparameter tuning for the classifier."""
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=3)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_classifier(X_train, y_train, classifier, param_grid):
    """Train a classifier on the extracted features with hyperparameter tuning."""
    best_classifier = tune_hyperparameters(X_train, y_train, classifier, param_grid)
    best_classifier.fit(X_train, y_train)
    return best_classifier

def test_classifier(best_classifier, X_test, y_test):
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{best_classifier.__class__.__name__} Classifier Accuracy: {accuracy:.4f}")
    report = "Classification Report:\n" + classification_report(y_test, y_pred)
    print(report)
    return report

def main():
    """Main function to extract, reduce, and classify image features."""
    train_directory = "data/reduced_train"
    test_directory = "data/reduced_test"

    image_sizes = [(4,4), (8,8), (12,12), (16,16)]
    classifiers = {
        'SVM': (SVC(), {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['linear', 'poly', 'rbf']}),
        'Random Forest': (RandomForestClassifier(n_estimators=100), {'n_estimators': [50, 100, 150, 200, 250]}),
        'Logistic Regression': (LogisticRegression(), {'max_iter': [500, 1000, 1500, 2000, 2500]}),
        'k-NN': (KNeighborsClassifier(n_neighbors=5), {'n_neighbors': [5, 10, 15, 20, 25]})
    }

    for image_size in image_sizes:
    
        X_train, y_train = extract_features_from_directory(train_directory, image_size)
        X_test, y_test = extract_features_from_directory(test_directory, image_size)

        # for n_components in [20, 40, 60, 80, 100]:
            # X_train = reduce_dimensionality(X_train, n_components=n_components)
            # X_test = reduce_dimensionality(X_test, n_components=n_components)

        for classifier_name in ['Random Forest', 'Logistic Regression']:
            classifier, param_grid = classifiers[classifier_name]
            # Train and evaluate classifier with tuning
            best_classifier = train_classifier(X_train, y_train, classifier, param_grid)
            report = test_classifier(best_classifier, X_test, y_test)
            with open(f'test_{classifier_name}_tiny_image_{image_size[0]}_{image_size[1]}.txt', 'w') as f:
                f.write(report)
    

if __name__ == "__main__":
    main()
