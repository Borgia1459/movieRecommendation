from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report

class SVM:
    def __init__(self):
        #we will use the wine dataset to explain the multiclass classification
        wine_data = load_wine()
        self.X = wine_data.data
        self.y = wine_data.target
        #print(f' input data size,X-shape: {self.X.shape},\n output data size,Y-shape: {self.y.shape}')
        #print(f' Label names: {wine_data.target_names}')
        #print(f' Feature names: {wine_data.feature_names}')
        n_class0,n_class1,n_class2 = len(self.y[self.y==0]),len(self.y[self.y==1]),len(self.y[self.y==2])
        print(f' class grade 0: {n_class0}, class grade 1: {n_class1}, class grade 2: {n_class2}') 
        

    def train(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
        model.fit(x_train, y_train)
        print(f' Training accuracy: {model.score(x_train, y_train)*100:.2f} %')

        pred=model.predict(x_test)
        print(f' Test accuracy:\n  {classification_report(y_test, pred)}')

    def predict(self, X):
        pass

class SVM3D:
    def __init__(self): 
        pass

if __name__ == '__main__':
    svm = SVM()
    #svm.__init__()
    svm.train(svm.X, svm.y)