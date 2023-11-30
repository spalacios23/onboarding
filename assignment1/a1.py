'''
HW: Satellite Image Classification
You may work on this assignment with ONE partner.

We use the SAT-4 Airborne Dataset: https://www.kaggle.com/datasets/crawford/deepsat-sat4.
Download "sat-4-full.mat" from Kaggle and place it in your working dir.

This dataset is large (500K satellite imgs of the Earth's surface).
Each img is 28x28 with 4 channels: red, green, blue, and NIR (near-infrared).
The imgs are labeled to the following 4 classes: 
barren land | trees | grassland | none

The MAT file from Kaggle contains 5 variables:
- annotations (explore this if you want to)
- train_x (400K training images), dim: (28, 28, 4, 400000)
- train_y (400k training labels), dim: (4, 400000)
- test_x (100K test images), dim: (28, 28, 4, 100000)
- test_y (100K test labels), dim: (4, 100000)

For inputs (train_x and test_x):
0th and 1st dim encode the row and column of pixel.
2nd dim describes the channel (RGB and NIR where R = 0, G = 1, B = 2, NIR = 3).
3rd dim encodes the index of the image.

Labels (train_y and test_y) are "one-hot encoded" (look this up).

Your task is to develop two classifiers, SVMs and MLPs, as accurate as you can.
'''

# TASK: Import libraries you need here
import joblib
import scipy.io
import numpy as np
import matplotlib as pltc
from sklearn import svm
from sklearn.neural_network import MLPClassifier





from joblib import dump, load # imported this for you, will be useful for workflow speed ups

# TASK: Load in the dataset
# Note: Use scipy.io.loadmat
# Note: Dealing with 400K and 100K images will take forever. 
# Feel free to train and test on small subsets (I did 10K and 2.5K, tune as you need).
# Just make sure your subset is rather uniformly distributed and not biased.
# Once you have your x_train, y_train, x_test, y_test variables (or however you name it),
# I suggest you save these variables using dump, then load them in subsequent runs.
# This will make things much faster as you wouldn't need to load in the full dataset each time.
"""
#load in the train_x variable from the sat-4-full.mat, getting 50,000 random images
train_x = scipy.io.loadmat('sat-4-full.mat')['train_x'][:,:,:,0:50000]
print("loaded train_x")
#load in the test_x variable from the sat-4-full.mat, getting 12,500 random images
test_x = scipy.io.loadmat('sat-4-full.mat')['test_x'][:,:,:,0:12500]
print("loaded test_x")
#load in the train_y variable from the sat-4-full.mat, getting 50,000 random labels
train_y = scipy.io.loadmat('sat-4-full.mat')['train_y'][:,0:50000]
print("loaded train_y")
#load in the test_y variable from the sat-4-full.mat, getting 12,500 random labels
test_y = scipy.io.loadmat('sat-4-full.mat')['test_y'][:,0:12500]

#dump the train_x, test_x, train_y, and test_y variables, whith unique names for file dump
dump(train_x, 'train_x.joblib') #train_x is an array of 400,000 images of size 28x28x4
print("dumped train_x")
dump(test_x, 'test_x.joblib')   #test_x is an array of 100,000 images of size 28x28x4
print("dumped test_x")
dump(train_y, 'train_y.joblib') #train_y is an array of 400,000 labels of size 4
print("dumped train_y")
dump(test_y, 'test_y.joblib')   #test_y is an array of 100,000 labels of size 4
print("dumped test_y")




"""
"""

train_x = load('train_x.joblib')
test_x = load('test_x.joblib')
train_y = load('train_y.joblib')
test_y = load('test_y.joblib')


#remove the 4th channel (NIR) of each train_x image
train_x = train_x[:,:,:3,:]
#remove the 4th channel (NIR) of each test_x image
test_x = test_x[:,:,:3,:]

# TASK: Pre-processing
# You need to figure out how to pass in the images as feature vectors to the models.
# You should not simply pass in the entire image as a flattened vector;
# otherwise, it's very slow and just not really effective
# Instead you should extract relevant features from the images.
# Refer to Section 4.1 of https://arxiv.org/abs/1509.03602, especially first three sentences
# and consider what features you want to extract
# And like the previous task, once you have your pre-processed feature vectors,
# you may want to dump and load because pre-processing will also take a while each time.
# MAKE SURE TO PRE-PROCESS YOUR TEST SET AS WELL!

#convert the rgb layers of each train_x image into hsv format
train_x_hsv = np.zeros((50000, 28, 28, 3))
for i in range(50000):
    train_x_hsv[i] = pltc.colors.rgb_to_hsv(train_x[:,:,:,i])

#convert the rgb layers of each test_x image into hsv format
test_x_hsv = np.zeros((12500, 28, 28, 3))
for i in range(12500):
    test_x_hsv[i] = pltc.colors.rgb_to_hsv(test_x[:,:,:,i])

print("finished converting to hsv")

#store the h layer of each train_x image into a new array
train_x_h = np.zeros((50000, 28, 28))
for i in range(50000):
    train_x_h[i] = train_x_hsv[i,:,:,0]

#store the s layer of each train_x image into a new array
train_x_s = np.zeros((50000, 28, 28))
for i in range(50000):
    train_x_s[i] = train_x_hsv[i,:,:,1]

#store the v layer of each train_x image into a new array
train_x_v = np.zeros((50000, 28, 28))
for i in range(50000):
    train_x_v[i] = train_x_hsv[i,:,:,2]

#store the h layer of each test_x image into a new array
test_x_h = np.zeros((12500, 28, 28))
for i in range(12500):
    test_x_h[i] = test_x_hsv[i,:,:,0]

#store the s layer of each test_x image into a new array
test_x_s = np.zeros((12500, 28, 28))
for i in range(12500):
    test_x_s[i] = test_x_hsv[i,:,:,1]

#store the v layer of each test_x image into a new array
test_x_v = np.zeros((12500, 28, 28))
for i in range(12500):
    test_x_v[i] = test_x_hsv[i,:,:,2]

print("finished storing hsv layers")
#take the standard deviation and mean of each h, s, and v layer of each train_x image and store them into a new array of size 6, and append the resulting array to a new array of size 10000
train_x_vector = []
for i in range(50000):
    temp_x_fv = []
    temp_x_fv.append(np.std(train_x_h[i]))
    temp_x_fv.append(np.mean(train_x_h[i]))
    temp_x_fv.append(np.std(train_x_s[i]))
    temp_x_fv.append(np.mean(train_x_s[i]))
    temp_x_fv.append(np.std(train_x_v[i]))
    temp_x_fv.append(np.mean(train_x_v[i]))
    train_x_vector.append(temp_x_fv)

print("finished train_x_vector")

#take the standard deviation and mean of each h, s, and v layer of each test_x image and store them into a new array of size 6, and append the resulting array to a new array of size 2500
test_x_vector = []
for i in range(12500):
    temp_x_fv = []
    temp_x_fv.append(np.std(train_x_h[i]))
    temp_x_fv.append(np.mean(train_x_h[i]))
    temp_x_fv.append(np.std(train_x_s[i]))
    temp_x_fv.append(np.mean(train_x_s[i]))
    temp_x_fv.append(np.std(train_x_v[i]))
    temp_x_fv.append(np.mean(train_x_v[i]))
    test_x_vector.append(temp_x_fv)

print("finished test_x_vector")

#convert the train_x_vector and test_x_vectors into numpy arrays
train_x_vector = np.asarray(train_x_vector)
test_x_vector = np.asarray(test_x_vector)


#dump the train_x_vector and test_x_vector feature vectors
dump(train_x_vector, 'train_x_vector.joblib') #train_x_vector is a feature vector 10000x6
dump(test_x_vector, 'test_x_vector.joblib')    #test_x_vector is a feature vector 2500x6
print("dumped train_x_vector and test_x_vector")
"""
"""
# TASK: TRAIN YOUR MODEL
# You have your feature vectors now, time to train.
# Again, train two models: SVM and MLP.
# Make them as accurate as possible. Tune your hyperparameters.
# Check for overfitting and other potential flaws as well.
"""




#load the train_x_vector and test_x_vector feature vectors
train_x_vector = load('train_x_vector.joblib')
test_x_vector = load('test_x_vector.joblib')
print("loaded train_x_vector and test_x_vector")

#load the train_y and test_y labels
train_y = load('train_y.joblib')
test_y = load('test_y.joblib')
print("loaded train_y and test_y")
"""
#find the best hyperparameters for the SVM model

best_score = 0
best_c = 0
best_gamma = 0
for c in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        clf = svm.SVC(kernel='rbf', C=c, gamma=gamma)
        clf.fit(train_x_vector, train_y[0])
        #print a status update when each combination of c and gamma is tested
        print("tested c: " + str(c) + " gamma: " + str(gamma))
        score = clf.score(test_x_vector, test_y[0])
        if score > best_score:
            best_score = score
            best_c = c
            best_gamma = gamma
print("best score: " + str(best_score))
print("best c: " + str(best_c))
print("best gamma: " + str(best_gamma))

#find the best hyperparameters for the MLP model

best_score = 0
best_hidden_layer_sizes = 0
best_max_iter = 0
for hidden_layer_sizes in [50, 100, 200]:
    for max_iter in [100, 200, 500]:
        clf2 = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        clf2.fit(train_x_vector, train_y[0])
        score = clf2.score(test_x_vector, test_y[0])
        if score > best_score:
            best_score = score
            best_hidden_layer_sizes = hidden_layer_sizes
            best_max_iter = max_iter
print("best score: " + str(best_score))
print("best hidden_layer_sizes: " + str(best_hidden_layer_sizes))
print("best max_iter: " + str(best_max_iter))


"""



#train the SVM model
clf = svm.SVC()
clf.fit(train_x_vector, train_y[0])
print("trained SVM model")

#train the MLP model
clf2 = MLPClassifier()
clf2.fit(train_x_vector, train_y[0])
print("trained MLP model")




# TASK: Visualizations
# Produce two visualizations, one for SVM and one for MLP.
# These should show the justifications for choosing your hyperparameters to your classifiers,
# such as kernel type, C value, gamma value, etc. for SVM or layer sizes, depths, itersm etc. for MLPs



#show the accuracy of the SVM model
print("SVM Accuracy:")
print(clf.score(test_x_vector, test_y[0]))


#show the accuracy of the MLP model
print("MLP Accuracy:")
print(clf2.score(test_x_vector, test_y[0]))



#produce a visualization for the SVM model



