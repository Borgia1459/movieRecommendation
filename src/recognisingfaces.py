import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class SVM:
    def __init__(self, degree=3, gamma='scale'):
        self.degree = degree
        self.gamma = gamma
        self.lfw_people = fetch_lfw_people(min_faces_per_person=80)
        self.X = self.lfw_people.data
        self.y = self.lfw_people.target

    def train(self):
        # Split the data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Define the parameter grid for GridSearchCV
        self.parameters = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }

        # Create an SVM classifier
        self.clf = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)

        # Perform grid search with cross-validation
        self.grid_search = GridSearchCV(self.clf, self.parameters, n_jobs=-1, cv=5)
        self.grid_search.fit(self.x_train, self.y_train)

        # Print the best parameters and the corresponding score
        print(f'Best parameters: {self.grid_search.best_params_}')
        self.clf_best= self.grid_search.best_estimator_
        self.pred=self.clf_best.predict(self.x_test)
        print(f'The average score: {self.clf_best.score(self.x_test, self.y_test)*100:.2f}%')
        
        print(f'Best cross-validation score: {self.grid_search.best_score_*100:.2f}%')

        # Evaluate the model on the test set
        self.y_pred = self.grid_search.predict(self.x_test)
        print(f'Test accuracy:\n{classification_report(self.y_test, self.y_pred, target_names=self.lfw_people.target_names)}')

        a
    def predict(self,):
        return self.grid_search.predict(self.X)
class PCAmodel:
    def __init__(self):
        self.lfw_people = fetch_lfw_people(min_faces_per_person=80)
        self.X = self.lfw_people.data
        self.y = self.lfw_people.target
        
    def train(self):
        #split the data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        #initialize the PCA model
        self.pca = PCA(n_components=150, whiten=True, random_state=42)
        self.svc= SVC(kernel='rbf', class_weight='balanced', random_state=42,C=1.0, gamma='scale')
        self.x_pca = self.pca.fit_transform(self.X)
        #pipeline the PCA model with the SVM model
        self.model= Pipeline(steps=[('pca', PCA(n_components=150, whiten=True, random_state=42)), ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42))])
        self.parameters_pipeline = {
            'svm__C': [1, 3, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__degree': [2, 3, 4],
            'svm__gamma': ['scale', 'auto']
        }
        self.grid_search = GridSearchCV(self.model, self.parameters_pipeline, n_jobs=1, cv=5)
        self.grid_search.fit(self.x_train, self.y_train)
        print(f'Best parameters:\n {self.grid_search.best_params_}')
        
if __name__ == '__main__':
    '''svm = SVM()
    svm.train()
    fig, ax = plt.subplots(3, 5)
    for i , axi in enumerate(ax.flat):
        axi.imshow(svm.X[i].reshape(62, 47), cmap='bone')
        axi.set(xticks=[], yticks=[], xlabel=svm.lfw_people.target_names[svm.y[i]])
    plt.show()'''
    pca = PCAmodel()
    pca.train()