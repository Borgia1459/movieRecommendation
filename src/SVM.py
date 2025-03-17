from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self):
        cancer_data = load_breast_cancer()
        self.X = cancer_data.data
        self.y = cancer_data.target
        #print(f' input data size,X-shape: {self.X.shape},\n output data size,Y-shape: {self.y.shape}')
        #print(f' Label names: {cancer_data.target_names}')
        #print(f' Feature names: {cancer_data.feature_names}')
        n_pos,n_neg = len(self.y[self.y==1]),len(self.y[self.y==0])
        print(f' Positive samples: {n_pos}, Negative samples: {n_neg}') 
        pass

    def train(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear',C=1.0,random_state=42)
        model.fit(x_train, y_train)
        print(f' Training accuracy: {model.score(x_train, y_train):.2f}')
        pass

    def predict(self, X):
        pass


if __name__ == '__main__':
    svm = SVM()
    #svm.__init__()
    svm.train(svm.X, svm.y)
    