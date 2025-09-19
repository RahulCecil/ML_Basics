### Decision Tree From Scratch ###

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
  

### We begin with a decision tree classifier implementation from scratch in Python ###
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'variety']
data = pd.read_csv("iris.csv", skiprows = 1, header = None, names = col_names)

#print(data.head(10))

class Node():
        ## Constructor Call ##
        def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None):
                ## For Decision Node ##
                self.feature_index = feature_index
                self.threshold = threshold
                self.left = left
                self.right = right
                self.info_gain = info_gain

                ## For Leaf Node ##
                self.value = value

class DecisionTreeClassifier():
        ## Constructor Call ##
        def __init__(self, min_samples_split = 2, max_depth = 2):

                ## Initialize root as None by default ##
                self.root = None

                ## Stopping Conditions ##
                self.min_samples_split = min_samples_split
                self.max_depth = max_depth
        
        ## Recursive function to build Tree ##
        def build_tree(self, dataset, curr_depth = 0):

                X, Y = dataset[:, :-1], dataset[:, -1]
                num_samples, num_features = np.shape(X)

                ## Split until stopping conditions ##
                if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
                        ## Find best split ##
                        best_split = self.get_best_split(dataset, num_samples, num_features)

                        ## check information gain ##
                        if best_split["info_gain"] > 0:
                                ## Recur Left ##
                                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)

                                ##Recur Right ##
                                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                                ## Return decision node ##
                                return Node(best_split["feature_index"], best_split["threshold"],
                                                left_subtree, right_subtree, best_split["info_gain"])

                        ## Compute Leaf Node ##
                        leaf_value = self.calculate_leaf_value(Y)
                        return Node(value = leaf_value)

        ## Function to finf best split ##
        def get_best_split(self, dataset, num_samples, num_features):

                ## Dictionary to store best split ##
                best_split = {}
                max_info_gain = -float("inf")

                ## Loop over all features ##
                for feature_index in range(num_features):
                        feature_values = dataset[:, feature_index]
                        possible_thresholds = np.unique(feature_values)

                        ## Loop over all thresholds ##
                        for threshold in possible_thresholds:
                                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                                ## Check if any split is empty ##
                                if len(dataset_left) > 0 and len(dataset_right) > 0:
                                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                                        ## Compute Information Gain ##
                                        curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

                                        ## Update best split if needed ##
                                        if curr_info_gain > max_info_gain:
                                                best_split["feature_index"] = feature_index
                                                best_split["threshold"] = threshold
                                                best_split["dataset_left"] = dataset_left
                                                best_split["dataset_right"] = dataset_right
                                                best_split["info_gain"] = curr_info_gain
                                                max_info_gain = curr_info_gain

                return best_split


        ## Function to split dataset ##
        def split(self, dataset, feature_index, threshold):
                
                dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
                dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])

                return dataset_left, dataset_right
        
        ## Function to compute Information Gain ##
        def information_gain(self, parent, l_child, r_child, mode = "entropy"):

                weight_l = len(l_child) / len(parent)
                weight_r = len(r_child) / len(parent)

                if mode == "gini":
                        gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
                
                else:
                        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
                
                return gain
        
        ## Function to compute entropy ##
        def entropy(self, y):

                class_labels = np.unique(y)
                entropy = 0

                for cls in class_labels:
                        p_cls = len(y[y == cls]) / len(y)
                        entropy += -p_cls * np.log2(p_cls)
                
                return entropy
        
        ## Function to compute Gini Index ##
        def gini_index(self, y):

                class_labels = np.unique(y)
                gini = 0

                for cls in class_labels:
                        p_cls = len(y[y == cls]) / len(y)
                        gini += p_cls ** 2
                
                return 1 - gini


        ## Function to compute leaf node ##
        def calculate_leaf_value(self, Y):
                
                Y = list(Y)
                return max(Y, key = Y.count)
        
        ## Function to print the tree ##
        def print_tree(self, tree = None, indent = 0):

                if tree is None:
                        tree = self.root

                ## If leaf node ##
                if tree is not None:
                        print(tree.value)

                else:
                        print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
                        print("%sleft:" % (indent), end="")
                        self.print_tree(tree.left, indent + indent)
                        print("%sright:" % (indent), end="")
                        self.print_tree(tree.right, indent + indent)

        ## Function to fit the tree ##
        def fit(self, X, Y):
                
                dataset = np.concatenate((X, Y), axis = 1)
                self.root = self.build_tree(dataset)
        
        ## Function to make predictions ##
        def predict(self, X):
                
                predictions = [self.make_prediction(x, self.root) for x in X]
                return predictions
        
        ## Function to make a single prediction ##
        def make_prediction(self, x, tree):

                ## If leaf node ##
                if tree is not None:
                        return tree.value
                
                feature_val = x[tree.feature_index]
                
                ## Traverse the tree ##
                if feature_val <= tree.threshold:
                        return self.make_prediction(x, tree.left)
                
                else:
                        return self.make_prediction(x, tree.right)
                

### Train - Test Split ###
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 41)

### Fit the model ###
classifier = DecisionTreeClassifier(min_samples_split = 3, max_depth = 3)
classifier.fit(X_train, Y_train)
#classifier.print_tree()

### Test the model ###
predictions = classifier.predict(X_test)
#accuracy_score(Y_test, predictions)