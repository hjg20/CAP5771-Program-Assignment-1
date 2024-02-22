import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        model = LogisticRegression(random_state=self.seed, max_iter=300)
        model.fit(Xtrain, ytrain)
        train_score = model.predict_proba(Xtrain)
        test_score = model.predict_proba(Xtest)
        ks = [1,2,3,4,5]
        train_scores = []
        test_scores = []
        for i in ks:
            train_scores.append(top_k_accuracy_score(ytrain, train_score, k=i))
            test_scores.append(top_k_accuracy_score(ytest, test_score, k=i))

        train_plot = plt.plot(ks, train_scores)
        test_plot = plt.plot(ks, test_scores)

        # Enter code and return the `answer`` dictionary

        answer = {
            'clf': model,
            'plot_k_vs_score_train': train_plot,
            'plot_k_vs_score_test': test_plot,
            'text_rate_accuracy_change': 'At each change in k, the slope of the curve representing the scores'
                                         'gets smaller and levels out. Therefore, the change in accuracy from'
                                         'k=1 to k=2 is greater than the change from k=4 to k=5.',
            'text_is_topk_useful_and_why': 'This metric is useful for this dataset because the accuracies seem to '
                                           'converge and both the training and testing lines on the graph are close'
                                           'together. This indicates that there is no significant over-fitting that is '
                                           'occurring and we get a great sense that the majority of our accuracy has '
                                           'occurred by k=3. Therefore, this metric is useful because it helps us '
                                           'understand out model better and the amount of guesses (k) we need to use. ',
            '1': {
                'score_train': train_scores[0],
                'score_test': test_scores[0]

            },
            '2': {
                'score_train': train_scores[1],
                'score_test': test_scores[1]

            },
            '3': {
                'score_train': train_scores[2],
                'score_test': test_scores[2]

            },
            '4': {
                'score_train': train_scores[3],
                'score_test': test_scores[3]

            },
            '5': {
                'score_train': train_scores[4],
                'score_test': test_scores[4]

            },
        }
        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  
    Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""

        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        nines = np.random.choice(np.where(ytrain == 9)[0],
                                 size=int(len(np.where(ytrain == 9)[0]) * 0.1),
                                 replace=False)
        test_nines = np.random.choice(np.where(ytest == 9)[0],
                                      size=int(len(np.where(ytest == 9)[0]) * 0.1),
                                      replace=False)
        sevens = np.where(ytrain == 7)[0]
        test_sevens = np.where(ytest == 7)[0]

        indices = np.concatenate((sevens, nines))
        test_indices = np.concatenate((test_sevens, test_nines))

        X = Xtrain[indices]
        y = ytrain[indices]
        Xtest = Xtest[test_indices]
        ytest = ytest[test_indices]

        y = np.where(y == 7, 0, y)
        y = np.where(y == 9, 1, y)
        ytest = np.where(ytest == 7, 0, ytest)
        ytest = np.where(ytest == 9, 1, ytest)

        # Answer is a dictionary with the same keys as part 1.B
        answer = {}

        answer["length_Xtrain"] = len(X)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = X.max()
        answer["max_Xtest"] = Xtest.max()

        return answer, X, y, Xtest, ytest
    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        svc = SVC(random_state=self.seed)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=svc, cv=KFold(n_splits=5))
        print(scores)

        answer = {
            'scores': {
                'mean_accuracy': None,
                'mean_recall': None,
                'mean_precision': None,
                'mean_f1': None,
                'std_accuracy': None,
                'std_recall': None,
                'std_precision': None,
                'std_f1': None
            },
            'cv': KFold(n_splits=5),
            'clf': svc,
            'is_precision_higher_than_recall': None,
            'explain_is_precision_higher_than_recall': None,
            'confusion_matrix_train': None,
            'confusion_matrix_test': None,

        }

        # Enter your code and fill the `answer` dictionary

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
