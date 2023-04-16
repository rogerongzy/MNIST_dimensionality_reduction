import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_mnist():
    with open('./train-images.idx3-ubyte', 'rb') as f1:
        train_image = np.frombuffer(f1.read(), np.uint8, offset=16).reshape(-1,28*28)
    with open('./train-labels.idx1-ubyte', 'rb') as f2:
        train_label = np.frombuffer(f2.read(), np.uint8, offset=8)
    with open('./t10k-images.idx3-ubyte', 'rb') as f3:
        test_image = np.frombuffer(f3.read(), np.uint8, offset=16).reshape(-1,28*28)
    with open('./t10k-labels.idx1-ubyte', 'rb') as f4:
        test_label = np.frombuffer(f4.read(), np.uint8, offset=8)

    image = np.concatenate([train_image, test_image])
    label = np.concatenate([train_label, test_label])
    
    image, label = shuffle(image, label, random_state=25)
    
    return image, label

    # showing the image and corresponding label
    # img = Image.fromarray(image_file[0].reshape(28,28)) # from PIL import Image
    # print(label_file[0])
    

def cross_validation_clf(image, label):
    clf_list = ['KNN', 'SGD', 'GNB', 'DT', 'SVC']
    # num_list = [0] # range(6)

    for index in range(5):
        if index == 0:
            clf = KNeighborsClassifier()
        elif index == 1:
            clf = SGDClassifier()
        elif index == 2:
            clf = GaussianNB()
        elif index == 3:
            clf = DecisionTreeClassifier()
        elif index == 4:
            clf = SVC(kernel='rbf')
        
        time_start = time.time()
        score = cross_val_score(clf, image, label) # none, scoring='precision_macro', scoring='recall_macro'
        time_end = time.time()
        print('--------------------------')
        print('classifier: ', clf_list[index])
        # print(score) # each fold
        print('score: ', np.mean(score))
        print('run time: ', time_end - time_start)
    

def PCA_dimension_reduction(image_input, ratio):
    pca = PCA(ratio) # maintain variance, 0.95-154
    pca.fit(image_input)
    image_reduced = pca.transform(image_input)
    image_recovered = pca.inverse_transform(image_reduced)	
    return image_reduced, image_recovered


def LDA_dimension_reduction(image_input, label_input, dimension):
    lda = LinearDiscriminantAnalysis(n_components=dimension) # maximum: categories - 1
    image_reduced = lda.fit_transform(image_input, label_input)
    return image_reduced


def PCA_trials(ratio):
    image_reduced, image_recovered = PCA_dimension_reduction(image, ratio)
    print(image_reduced.shape)
    cross_validation_clf(image_reduced, label)
    

def LDA_trials(dimension):
    image_reduced = LDA_dimension_reduction(image, label, dimension)
    print(image_reduced.shape)
    cross_validation_clf(image_reduced, label)


if __name__ == '__main__':
    image, label = load_mnist()
    cross_validation_clf(image, label)

    # PCA_trials(0.75)
    # LDA_trials(9)