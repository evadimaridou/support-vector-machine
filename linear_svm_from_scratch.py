import numpy as np
import keras
import time

class SVM:
    
    def __init__(self, my_lambda, eta, epochs):
        self.my_lambda = my_lambda
        self.epochs = epochs
        self.eta = eta
    
    def fit(self, train_data, train_labels):
        self.weights = np.zeros(train_data.shape[1])
        self.bias = 0

        for _ in range (self.epochs):
            for idx, sample in enumerate(train_data):
                condition = train_labels[idx]*(np.dot(self.weights, sample) + self.bias)

                if (condition >= 1):
                    self.gradient_descent(True, train_labels[idx], sample)
                else:
                    self.gradient_descent(False, train_labels[idx], sample)
                    
    def gradient_descent(self, condition, label, data):
        if(condition):
            self.weights -= self.eta*2*self.my_lambda*self.weights
        else:
            self.weights -= self.eta*(2*self.my_lambda*self.weights - np.dot(data, label))
            self.bias -= self.eta*label

    def predict(self, test_data):
        self.test_data = test_data
        classif = np.dot(self.test_data, self.weights) + self.bias
        return np.sign(classif)
    
    def accuracy(self, test_labels, accuracy_type):
        accuracy = 0
        for idx, sample in enumerate(self.test_data):
            if (my_svm.predict(sample) == test_labels[idx]):
                accuracy += 1
        if (accuracy_type == "test"):
            print(f"Test accuracy is: {accuracy/test_labels.shape[0]}")
        elif(accuracy_type == "training"):
            print(f"Training accuracy is: {accuracy/test_labels.shape[0]}")

if __name__ == "__main__":
    
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()
    train_data = train_data.reshape(-1, 3072)
    test_data = test_data.reshape(-1, 3072)

    class1, class2 = (0, 9)

    print(f"Classes are {class1} and {class2}:")

    train_labels = np.squeeze(train_labels)
    train_labels = np.array(train_labels, dtype = np.int32)
    test_labels = np.squeeze(test_labels)
    test_labels = np.array(test_labels, dtype = np.int32)


    # Filter training data
    indices_train = np.where((train_labels == class1) | (train_labels == class2))[0]
    train_data = train_data[indices_train]
    train_labels = train_labels[indices_train]

    # Filter test data
    indices_test = np.where((test_labels == class1) | (test_labels == class2))[0]
    test_data = test_data[indices_test]
    test_labels = test_labels[indices_test]

    train_labels[train_labels == class1] = 1
    test_labels[test_labels == class1] = 1

    train_labels[train_labels == class2] = -1
    test_labels[test_labels == class2] = -1
        
    #for numerical stability
    train_data = train_data / 255
    test_data = test_data / 255

    start = time.time()
            
    my_svm = SVM(my_lambda = 1e-3, eta = 1e-4, epochs = 25)
    my_svm.fit(train_data, train_labels)
    my_svm.predict(train_data)
    my_svm.accuracy(train_labels, "training")
    my_svm.predict(test_data)
    my_svm.accuracy(test_labels, "test")
    end = time.time()
    my_time = (end - start)/60
    print(f"Time taken: {my_time}")


