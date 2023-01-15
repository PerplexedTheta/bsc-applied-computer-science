# city3114_proc1, jake deery, 2020

# 'include' i'm a cpp developer and its late okay, let me write this like a goddamn conapp
from sklearn import datasets
from sklearn import neighbors
from sklearn import linear_model



# i know python supports globally scoped code but thats utterly rancid even traditional c doesnt do that
class main():
    def doLearning(self, passedName, passedKnn, passedLogistic, passedData, passedGroundTruth):
        # vars
        name = passedName
        knn = passedKnn
        logistic = passedLogistic
        data = passedData
        groundTruth = passedGroundTruth

        #
        # do some prep
        # ... prior to the actual learning algorithm
        data = data / data.max()  # divide the list's contents by the largest value within
        nSamples = len(data)  # fetches the total length of the 'array' i know its a list okaaaaaaaay!

        trainData = data[:int(.9 * nSamples)]  # take the first 90% of the data as our set
        trainGroundTruth = groundTruth[:int(.9 * nSamples)]  # '  '
        testData = data[int(.9 * nSamples):]  # take the last 10% of the data as our set
        testGroundTruth = groundTruth[int(.9 * nSamples):]  # '  '

        #
        # calculate the knn fit
        # ... using the knn object passed by the parent
        calculatedFit = knn.fit(trainData, trainGroundTruth)\
            .score(testData, testGroundTruth) # do the knn magic
        calculatedLogistic = logistic.fit(trainData, trainGroundTruth)\
            .score(testData, testGroundTruth) # do the logistic magic

        #
        # print the final results
        # i will discuss more on what is happening here within my report
        print('[I] KNN score of %s (percentage): %f' % (name, calculatedFit))
        print('[I] Logistic Regression score (percentage): %f' % (calculatedLogistic))
        print('')



    def __init__(self): # why is this syntax a thing
        #
        # vars
        # global scoping is gross so this crap goes here
        digitsData, digitsGroundTruth = datasets.load_digits(return_X_y=True)
        irisData, irisGroundTruth = datasets.load_iris(return_X_y=True)
        wineData, wineGroundTruth = datasets.load_wine(return_X_y=True) # load some sklearn datasets for juicy preloaded
        # goodness ayyyy thank based jesus for sklearn, also that bool splits the dataset into data & target

        #
        # init stuff
        # say aite, innit (get it?)
        print('[I] CITY3114_PROC1, Jake Deery, 2020') # what chimpanzee decided not requiring a ; was a good idea
        print('') # what chimpanzee decided not requiring a \n was a good idea

        #
        # set up the learning algos
        # this will change before each process run
        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree',
                                             leaf_size=30, p=2)
        # this is the classifier which should do some magic for us which i will of course fully explain and know exactly
        # what i mean, totally, in my report

        logistic = linear_model.LogisticRegression(max_iter=1000)  # a form of linear regression. this will simply act
        # as a comparison against the knn -- to see if the values are close which they are because i am a god

        #
        # now call the function
        # ... to do the repeatable work
        print('[W] Changing to digits dataset . . . done')
        print('[I] Dataset info available:' +
              'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html')
        print('')

        self.doLearning('Digits Data, 5, uniform, ball_tree, 30, 2', knn, logistic, digitsData, digitsGroundTruth)

        #
        # trying different knn vars on digits
        # ... to assess different results
        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree',
                                             leaf_size=30, p=2)
				# flag 1: n_neighbors = number of neighbours to classify with a point
				# flag 2: weights = how to weight local neighbourhood, uniformly or by inverse of distance
				# flag 3: algorithm = algorithm to optimise function, either ball_tree or kd_tree
				# flag 4: leaf_size = number of leaves in the tree, optimal value varies by dataset
				# flag 5: p = power against metric, where 1 is manhattan_distance and 2 is euclidean_distance
        self.doLearning('Digits Data, 5, distance, ball_tree, 30, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Digits Data, 5, uniform, kd_tree, 30, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Digits Data, 5, distance, kd_tree, 30, 2', knn, logistic, digitsData, digitsGroundTruth)

        #
        # no change in the result should be seen
        # lets try changing numbers
        print('[I] No change in output expected . . . Altering n_neightbours, leaf_size, p')
        knn = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='ball_tree',
                                             leaf_size=20, p=2)
        self.doLearning('Digits Data, 4, uniform, ball_tree, 20, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='ball_tree',
                                             leaf_size=10, p=2)
        self.doLearning('Digits Data, 3, uniform, ball_tree, 10, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree',
                                             leaf_size=40, p=2)
        self.doLearning('Digits Data, 6, uniform, ball_tree, 40, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='ball_tree',
                                             leaf_size=50, p=2)
        self.doLearning('Digits Data, 7, uniform, ball_tree, 50, 2', knn, logistic, digitsData, digitsGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='ball_tree',
                                             leaf_size=50, p=3)
        self.doLearning('Digits Data, 7, uniform, ball_tree, 50, 3', knn, logistic, digitsData, digitsGroundTruth)

        #
        # okay moving onto dataset two
        # same again, try picking a different algo before going onto number changing
        print('[W] Changing to iris dataset . . . done')
        print('[I] Dataset info available:' +
              'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html')
        print('')

        self.doLearning('Iris Data, 5, uniform, ball_tree, 30, 2', knn, logistic, irisData, irisGroundTruth)

        #
        # trying different knn vars on iris
        # ... to assess different results
        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Iris Data, 5, distance, ball_tree, 30, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Iris Data, 5, uniform, kd_tree, 30, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Iris Data, 5, distance, kd_tree, 30, 2', knn, logistic, irisData, irisGroundTruth)

        #
        # no change in the result should be seen
        # lets try changing numbers
        print('[I] Picking distance, ball_tree . . . Altering n_neightbours, leaf_size, p')
        knn = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='ball_tree',
                                             leaf_size=20, p=2)
        self.doLearning('Iris Data, 4, uniform, ball_tree, 20, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='ball_tree',
                                             leaf_size=10, p=2)
        self.doLearning('Iris Data, 3, uniform, ball_tree, 10, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='ball_tree',
                                             leaf_size=40, p=2)
        self.doLearning('Iris Data, 6, uniform, ball_tree, 40, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree',
                                             leaf_size=50, p=2)
        self.doLearning('Iris Data, 7, uniform, ball_tree, 50, 2', knn, logistic, irisData, irisGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree',
                                             leaf_size=50, p=3)
        self.doLearning('Iris Data, 7, uniform, ball_tree, 50, 3', knn, logistic, irisData, irisGroundTruth)

        #
        # okay moving onto dataset three
        # same again, try picking a different algo before going onto number changing
        print('[W] Changing to wine dataset . . . done')
        print('[I] Dataset info available:' +
              'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html')
        print('')

        self.doLearning('Wine Data, 5, uniform, ball_tree, 30, 2', knn, logistic, wineData, wineGroundTruth)

        #
        # trying different knn vars on wine
        # ... to assess different results
        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Wine Data, 5, distance, ball_tree, 30, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Wine Data, 5, uniform, kd_tree, 30, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree',
                                             leaf_size=30, p=2)
        self.doLearning('Wine Data, 5, distance, kd_tree, 30, 2', knn, logistic, wineData, wineGroundTruth)

        #
        # no change in the result should be seen
        # lets try changing numbers
        print('[I] Picking uniform, ball_tree . . . Altering n_neightbours, leaf_size, p')
        knn = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='ball_tree',
                                             leaf_size=20, p=2)
        self.doLearning('Wine Data, 4, uniform, ball_tree, 20, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='ball_tree',
                                             leaf_size=10, p=2)
        self.doLearning('Wine Data, 3, uniform, ball_tree, 10, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree',
                                             leaf_size=40, p=2)
        self.doLearning('Wine Data, 6, uniform, ball_tree, 40, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='ball_tree',
                                             leaf_size=50, p=2)
        self.doLearning('Wine Data, 7, uniform, ball_tree, 50, 2', knn, logistic, wineData, wineGroundTruth)

        knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='ball_tree',
                                             leaf_size=50, p=3)
        self.doLearning('Wine Data, 7, uniform, ball_tree, 50, 3', knn, logistic, wineData, wineGroundTruth)



    def __del__(self):
        # taaaaraaaaaaaa!
        print('[E] Bye!')
        print('')


#
#
# run class & delet
doProc = main()
del doProc

#
# python sucks
# bye
