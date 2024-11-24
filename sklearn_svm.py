import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
import keras
import time

if __name__ == "__main__":
        
    class1, class2 = (0, 9)

    (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()
    train_data = train_data.reshape(-1, 3072)
    test_data = test_data.reshape(-1, 3072)

    train_labels = np.squeeze(train_labels)
    train_labels = np.array(train_labels, dtype = np.int32)
    test_labels = np.squeeze(test_labels)
    test_labels = np.array(test_labels, dtype = np.int32)
        
    print(f"Classes are {class1} and {class2}:")
        
    #airplane 0, truck 9, cat 3, dog 5

    #Filter training data
    indices_train = np.where((train_labels == class1) | (train_labels == class2))[0]
    train_data = train_data[indices_train]
    train_labels = train_labels[indices_train]

    #Filter test data
    indices_test = np.where((test_labels == class1) | (test_labels == class2))[0]
    test_data = test_data[indices_test]
    test_labels = test_labels[indices_test]

    train_labels[train_labels == class1] = 1
    test_labels[test_labels == class1] = 1

    train_labels[train_labels == class2] = -1
    test_labels[test_labels == class2] = -1
    train_data = train_data / 255
    test_data = test_data / 255
        
    start = time.time()
    svc = SVC(kernel='poly', C = 1, degree = 2, coef0 = 1).fit(train_data, train_labels)
    y_pred = svc.predict(train_data) 
    accuracy = accuracy_score(train_labels, y_pred) 
    print('Training Accuracy:', accuracy)
    y_pred = svc.predict(test_data) 
    accuracy = accuracy_score(test_labels, y_pred) 
    print('Test Accuracy:', accuracy) 
    end = time.time()
    my_time = (end - start)/60
    print(f"Time taken: {my_time}")

    


