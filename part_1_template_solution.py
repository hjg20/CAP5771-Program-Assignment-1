# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.cv = None
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        tree = DecisionTreeClassifier(random_state=self.seed)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=tree, cv=KFold(n_splits=5))

        answer = {}
        answer["clf"] = tree  # the estimator (classifier instance)
        answer["cv"] = KFold(n_splits=5)  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        answer["scores"] = scores
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        tree = DecisionTreeClassifier(random_state=self.seed)
        self.cv = ShuffleSplit(n_splits=5, random_state=self.seed)
        scores = cross_validate(tree, X, y, cv=self.cv)

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {}
        answer["clf"] = tree
        answer["cv"] = ShuffleSplit(n_splits=5, random_state=self.seed)
        answer["scores"] = scores
        answer["explain_kfold_vs_shuffle_split"] = ("The pros of k-fold cross validation is that every data point "
                                                    "is used for validation once which reduces the risk of bias "
                                                    "in the model. The cons are that since the folds aren't "
                                                    "randomly shuffled, there is a chance that, if the data has "
                                                    "order, the folds don't represent the whole dataset. The pros "
                                                    "of Shuffle-Split is that the randomization that occurs allows"
                                                    "for the folds to represent more of the data and not be prone"
                                                    "to an ordered dataset. The cons are that each run can produce"
                                                    "different results.")
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`
        splits = [2, 5, 8, 16]
        tree = DecisionTreeClassifier(random_state=self.seed)

        answer = {}
        answer[2] = {}
        answer[5] = {}
        answer[8] = {}
        answer[16] = {}
        for i in splits:

            scores = cross_validate(tree, X, y, cv=ShuffleSplit(n_splits=i, random_state=self.seed))

            answer[i]["clf"] = tree
            answer[i]["cv"] = ShuffleSplit(n_splits=i, random_state=self.seed)
            answer[i]["scores"] = scores

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        model = RandomForestClassifier(random_state=self.seed)
        tree = DecisionTreeClassifier(random_state=self.seed)
        rfscores = cross_validate(model, X, y, cv=self.cv)
        treescores = cross_validate(tree, X, y, cv=self.cv)

        rfmean_fit_time = rfscores['fit_time'].mean()
        rfstd_fit_time = rfscores['fit_time'].std()
        rfmean_accuracy = rfscores['test_score'].mean()
        rfstd_accuracy = rfscores['test_score'].std()
        rfvar = rfstd_accuracy ** 2

        tmean_fit_time = treescores['fit_time'].mean()
        tstd_fit_time = treescores['fit_time'].std()
        tmean_accuracy = treescores['test_score'].mean()
        tstd_accuracy = treescores['test_score'].std()
        tvar = tstd_accuracy **2

        # Find max scoring model
        if rfmean_accuracy > tmean_accuracy:
            max_acc_model = "Random Forest"
        elif rfmean_accuracy > tmean_accuracy:
            max_acc_model = "Decision Tree"
        else:
            max_acc_model = "Both"

        # Find the lowest variance model
        if rfvar > tvar:
            min_var_model = "Random Forest"
        elif rfvar > tvar:
            min_var_model = "Decision Tree"
        else:
            min_var_model = "Both"

        # Find the fastest model
        if rfmean_fit_time > tmean_fit_time:
            fastest_model = "Random Forest"
        elif rfmean_fit_time > tmean_fit_time:
            fastest_model = "Decision Tree"
        else:
            fastest_model = "Both"

        answer = {
            "clf_RF": model,
            "clf_DT": tree,
            "cv": self.cv,
            "scores_RF": {
                "mean_fit_time": rfmean_fit_time,
                "std_fit_time": rfstd_fit_time,
                "mean_accuracy": rfmean_accuracy,
                "std_accuracy": rfstd_accuracy
            },
            "scores_DT": {
                "mean_fit_time": tmean_fit_time,
                "std_fit_time": tstd_fit_time,
                "mean_accuracy": tmean_accuracy,
                "std_accuracy": tstd_accuracy
            },
            "model_highest_accuracy": max_acc_model,
            "model_lowest_variance": min_var_model,
            "model_fastest": fastest_model
        }

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """


        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X, y)
        y_pred = model.predict(X)
        yt_pred = model.predict(Xtest)

        estimators = [10, 50, 100, 200]
        depths = [3, 7, 11]

        params = {
            "max_depth": depths,
            "n_estimators": estimators
        }

        gscv = GridSearchCV(estimator=model, param_grid=params, cv=ShuffleSplit(n_splits=5, random_state=self.seed))
        gscv.fit(X, y)
        y_pred_best = gscv.best_estimator_.predict(X)
        yt_pred_best = gscv.best_estimator_.predict(Xtest)

        answer = {
            "clf": model,
            "default_parameters": {
                "random_state": self.seed
            },
            "best_estimator": gscv.best_params_,
            "grid_search": gscv,
            "mean_accuracy_cv": gscv.best_score_,
            "confusion_matrix_train_orig": confusion_matrix(y, y_pred),
            "confusion_matrix_train_best": confusion_matrix(y, y_pred_best),
            "confusion_matrix_test_orig": confusion_matrix(ytest, yt_pred),
            "confusion_matrix_test_best": confusion_matrix(ytest, yt_pred_best),
            "accuracy_orig_full_training": np.trace(confusion_matrix(y, y_pred)) / np.sum(confusion_matrix(y, y_pred)),
            "accuracy_best_full_training": np.trace(confusion_matrix(y, y_pred_best)) / np.sum(confusion_matrix(y, y_pred_best)),
            "accuracy_orig_full_testing": np.trace(confusion_matrix(ytest, yt_pred)) / np.sum(confusion_matrix(ytest, yt_pred)),
            "accuracy_best_full_testing": np.trace(confusion_matrix(ytest, yt_pred_best)) / np.sum(confusion_matrix(ytest, yt_pred_best)),

        }

        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """
        return answer
