import numpy as np
import keras
from cvxopt import matrix, solvers
import time

class SVM:

    def __init__(self, kernel="linear", r = 1, d = 2, gamma = 10, C = 1):
        self.kernel = kernel
        self.r = r
        self.d = d
        self.gamma = gamma
        self.C = C
    
    def compute_kernel(self, u, v):
        if (self.kernel=="linear"):
            kernel = np.dot(u, v.T)

        elif (self.kernel == "poly"):
            kernel = np.power((np.dot(u, v.T) + self.r), self.d)

        elif (self.kernel =="rbf"):
            kernel = -self.gamma*np.dot(u - v, u - v)
            kernel = np.exp(kernel)

        return kernel
    
    def construct_kernel_matrix(self):
        num_samples = self.train_data.shape[0]
        self.my_kernel = np.zeros((num_samples, num_samples))
        
        for i, sample_1 in enumerate(self.train_data):
            for j, sample_2 in enumerate(self.train_data):
                self.my_kernel[i][j] = self.compute_kernel(sample_1, sample_2)
        
        self.P = np.dot(self.train_labels, self.train_labels.T) * self.my_kernel

        self.P += 1e-10*np.eye(self.P.shape[0])

        self.P_matrix = matrix(self.P)

    def construct_qp_param(self):
        num_samples = self.train_data.shape[0]
        
        self.linear_coef = matrix(np.ones(num_samples)*(-1))
        
        self.A = matrix(self.train_labels * 1.0, (1, num_samples))
       
        self.b = matrix(0.0)
    
        self.G = matrix(np.vstack((np.eye(num_samples) * (-1), np.eye(num_samples))))
        self.h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.C)))

        self.init_values = matrix(np.zeros((num_samples, 1)))

    def quadr_program(self):
        solvers.options['show_progress'] = False
        solution = solvers.qp(P = self.P_matrix, q = self.linear_coef, G = self.G, h = self.h, A = self.A, b = self.b, 
                              initvals=self.init_values)
        
        self.all_alphas = np.array(solution['x'])
        self.is_sv = np.where(np.logical_and(self.all_alphas > 0, self.all_alphas <= self.C))[0]    #indexes of the support vectors alpha
        
        #all the data that are support vectors
        self.sv = self.train_data[self.is_sv]

        #labels of support vectors
        self.sv_y = self.train_labels[self.is_sv]
        self.quadr_coef_ind = self.P[self.is_sv]

        #alphas of support vectors
        self.alphas = self.all_alphas[self.is_sv]

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_labels = train_labels.reshape(-1, 1).astype(np.double)

        self.construct_kernel_matrix()
        self.construct_qp_param()
        self.quadr_program()
        self.compute_bias()

    def compute_bias(self):
        #select one of the support vectors
        self.margin_sv = self.is_sv[0]  

        #find label of said support vector
        ys = self.train_labels[self.margin_sv]

        #compute the bias
        self.bias = ys - np.sum(self.alphas * self.sv_y * self.my_kernel[self.is_sv, self.margin_sv].reshape(-1, 1))

    def predict(self, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_labels = self.test_labels.reshape(-1, 1).astype(np.double)

        quadr_coef = np.zeros((self.sv.shape[0], self.test_data.shape[0]))
        for i, sample_1 in enumerate(self.sv):
            for j, sample_2 in enumerate(self.test_data):
                quadr_coef[i][j] = self.compute_kernel(sample_1, sample_2)
        
        #it creates an array and every column represents one of the test samples' sums
        #the np.sum resulting array is (num of support vectors x num of test samples). Every column is the corresponding sum.
        self.predicted_labels = np.sign(np.sum(self.alphas * self.sv_y * quadr_coef, axis=0) + self.bias)

    def accuracy(self, type_accuracy):
        accuracy = 0
        for idx, _ in enumerate(self.test_data):
            if (self.predicted_labels[idx]) == self.test_labels[idx]:
                accuracy += 1
        if (type_accuracy == "test"):
                print(f"Test accuracy is: {accuracy/self.test_data.shape[0]}")
        elif (type_accuracy == "train"):
                print(f"Training accuracy is: {accuracy/self.test_data.shape[0]}")
        
if __name__ =="__main__":
    
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()
    train_data = train_data.reshape(-1, 3072)
    test_data = test_data.reshape(-1, 3072)

    train_labels = np.squeeze(train_labels)
    train_labels = np.array(train_labels, dtype = np.int32)
    test_labels = np.squeeze(test_labels)
    test_labels = np.array(test_labels, dtype = np.int32)
    
    class1, class2 = (0, 9)
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
    np.random.seed(1)
    
    #for numerical stability
    train_data = train_data / 255
    test_data = test_data / 255
        
            
    start = time.time()
    my_svm = SVM("rbf", gamma = 0.01, C = 1)
    my_svm.fit(train_data, train_labels)
    my_svm.predict(train_data, train_labels)
    my_svm.accuracy("train")
    my_svm.predict(test_data, test_labels)
    my_svm.accuracy("test")
    end = time.time()
    my_time = (end - start)/60
    print(f"Time taken: {my_time}")
    
    
    