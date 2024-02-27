# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.
import utils as u
import new_utils as nu
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score, confusion_matrix

import numpy as np
from numpy.typing import NDArray
from typing import Any


# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
            self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {
            "nb_classes_train": len(np.unique(ytrain)),
            "nb_classes_test": len(np.unique(ytest)),
            "class_count_train": np.unique(ytrain, return_counts=True),
            "class_count_test": np.unique(ytest, return_counts=True),
            "length_Xtrain": len(Xtrain),
            "length_Xtest": len(Xtest),
            "length_ytrain": len(ytrain),
            "length_ytest": len(ytest),
            "max_Xtrain": Xtrain.max(),
            "max_Xtest": Xtest.max()

        }
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        # Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        # ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
            self,
            X: NDArray[np.floating],
            y: NDArray[np.int32],
            Xtest: NDArray[np.floating],
            ytest: NDArray[np.int32],
            ntrain_list: list[int] = [],
            ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """

        answer = {
            1000: {
                "partC": None,
                "partD": None,
                "partF": None,
                "ntrain": None,
                "ntest": None,
                "class_count_train": None,
                "class_count_test": None,
            },

            5000: {
                "partC": None,
                "partD": None,
                "partF": None,
                "ntrain": None,
                "ntest": None,
                "class_count_train": None,
                "class_count_test": None,
            },
            10000: {
                "partC": None,
                "partD": None,
                "partF": None,
                "ntrain": None,
                "ntest": None,
                "class_count_train": None,
                "class_count_test": None,
            }
        }

        for i in range(len(ntrain_list)):
            Xtrain = X[0:ntrain_list[i], :]
            ytrain = y[0:ntrain_list[i]]
            Xtest = X[ntrain_list[i]:ntrain_list[i] + ntest_list[i]]
            ytest = y[ntrain_list[i]:ntrain_list[i] + ntest_list[i]]

            tree = DecisionTreeClassifier(random_state=self.seed)
            scores = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=tree, cv=KFold(n_splits=5))

            answer[ntrain_list[i]]["partC"] = {
                'clf': tree,
                'cv': KFold(n_splits=5),
                'scores': scores
            }

            tree = DecisionTreeClassifier(random_state=self.seed)
            cv = ShuffleSplit(n_splits=5, random_state=self.seed)
            scores = cross_validate(tree, Xtrain, ytrain, cv=cv)

            answer[ntrain_list[i]]["partD"] = {
                'clf': tree,
                'cv': ShuffleSplit(n_splits=5, random_state=self.seed),
                'scores': scores,
                "explain_kfold_vs_shuffle_split": ("The pros of k-fold cross validation is that every data point "
                                                   "is used for validation once which reduces the risk of bias "
                                                   "in the model. The cons are that since the folds aren't "
                                                   "randomly shuffled, there is a chance that, if the data has "
                                                   "order, the folds don't represent the whole dataset. The pros "
                                                   "of Shuffle-Split is that the randomization that occurs allows"
                                                   "for the folds to represent more of the data and not be prone"
                                                   "to an ordered dataset. The cons are that each run can produce"
                                                   "different results.")
            }

            model = LogisticRegression(random_state=self.seed, max_iter=300)
            # tree = DecisionTreeClassifier(random_state=self.seed)
            # rfscores = cross_validate(model, X, y, cv=cv)
            # treescores = cross_validate(tree, X, y, cv=cv)
            #
            # rfmean_fit_time = rfscores['fit_time'].mean()
            # rfstd_fit_time = rfscores['fit_time'].std()
            # rfmean_accuracy = rfscores['test_score'].mean()
            # rfstd_accuracy = rfscores['test_score'].std()
            # rfvar = rfstd_accuracy ** 2
            #
            # tmean_fit_time = treescores['fit_time'].mean()
            # tstd_fit_time = treescores['fit_time'].std()
            # tmean_accuracy = treescores['test_score'].mean()
            # tstd_accuracy = treescores['test_score'].std()
            # tvar = tstd_accuracy ** 2
            #
            # # Find max scoring model
            # if rfmean_accuracy > tmean_accuracy:
            #     max_acc_model = "Logistic Regression"
            # elif rfmean_accuracy > tmean_accuracy:
            #     max_acc_model = "Decision Tree"
            # else:
            #     max_acc_model = "Both"
            #
            # # Find the lowest variance model
            # if rfvar > tvar:
            #     min_var_model = "Logistic Regression"
            # elif rfvar > tvar:
            #     min_var_model = "Decision Tree"
            # else:
            #     min_var_model = "Both"
            #
            # # Find the fastest model
            # if rfmean_fit_time > tmean_fit_time:
            #     fastest_model = "Logistic Regression"
            # elif rfmean_fit_time > tmean_fit_time:
            #     fastest_model = "Decision Tree"
            # else:
            #     fastest_model = "Both"

            # answer[ntrain_list[i]]["partF"] = {
            #     "clf_RF": model,
            #     "clf_DT": tree,
            #     "cv": cv,
            #     "scores_RF": {
            #         "mean_fit_time": rfmean_fit_time,
            #         "std_fit_time": rfstd_fit_time,
            #         "mean_accuracy": rfmean_accuracy,
            #         "std_accuracy": rfstd_accuracy
            #     },
            #     "scores_DT": {
            #         "mean_fit_time": tmean_fit_time,
            #         "std_fit_time": tstd_fit_time,
            #         "mean_accuracy": tmean_accuracy,
            #         "std_accuracy": tstd_accuracy
            #     },
            #     "model_highest_accuracy": max_acc_model,
            #     "model_lowest_variance": min_var_model,
            #     "model_fastest": fastest_model
            # }

            scores = cross_validate(model, X, y, cv=cv, return_train_score=True)
            model.fit(X, y)
            scores_train_F = model.score(X, y)
            scores_test_F = model.score(Xtest, ytest)  # scalar
            mean_cv_accuracy_F = scores["test_score"].mean()

            answer[ntrain_list[i]]["partF"] = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": model,
                "cv": cv,
                "conf_mat_train": confusion_matrix(y, model.predict(X)),
                "conf_mat_test": confusion_matrix(ytest, model.predict(Xtest)),
            }

            answer[ntrain_list[i]]['ntrain'] = ntrain_list[i]
            answer[ntrain_list[i]]['ntest'] = ntest_list[i]

            answer[ntrain_list[i]]['class_count_train'] = len(np.unique(y[0:ntrain_list[i]]))
            answer[ntrain_list[i]]['class_count_test'] = len(np.unique(y[ntrain_list[i]:ntrain_list[i] + ntest_list[i]]))

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        print(answer)
        return answer
