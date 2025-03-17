import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from config import TRAIN
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class Cardiotography:
    def __init__(self):
        try:
            df = fetch_ucirepo(id=12)
            x = df.data.features.values
            y = df.data.targets.values.ravel()  # Ensure y is a 1D array
            print(f'Input data size, X-shape: {x.shape},\nOutput data size, Y-shape: {y.shape}')
            self.X = x
            self.y = y
        except FileNotFoundError:
            print(f"File not found: {TRAIN}")
        except Exception as e:
            print(f"An error occurred while fetching the dataset: {e}")

    def train(self):
        try:
            # Ensure the number of samples in X and y match
            if self.X.shape[0] != self.y.shape[0]:
                raise ValueError(f"Inconsistent number of samples: X has {self.X.shape[0]} samples, y has {self.y.shape[0]} samples")

            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Define the parameter grid for GridSearchCV
            parameters = {
                'C': [100, 1e3, 1e4, 1e5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }

            # Create an SVM classifier
            clf = SVC(class_weight='balanced', random_state=42)

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
            grid_search.fit(x_train, y_train)

            # Print the best parameters and the corresponding score
            print(f'Best parameters: {grid_search.best_params_}')
            clf_best = grid_search.best_estimator_
            pred = clf_best.predict(x_test)
            print(f'The average score: {clf_best.score(x_test, y_test) * 100:.2f}%')
            
            print(f'Best cross-validation score: {grid_search.best_score_ * 100:.2f}%')
            report = classification_report(y_test, pred)
            print(f'Classification report:\n{report}')

            # Evaluate the model on the test set
            y_pred = grid_search.predict(x_test)
            print(f'Test accuracy:\n{classification_report(y_test, y_pred)}')
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def predict(self, X):
        try:
            return self.grid_search.predict(X)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

if __name__ == '__main__':
    try:
        ctg = Cardiotography()
        ctg.train()
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")