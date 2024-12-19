from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


# Define the parameter grid
def SVM_parameters(X, y):
    param_grid = {
        'C': [0.1, 1, 10],               # Regularization parameter
        'kernel': ['linear', 'rbf'],     # Kernel types to try
        'gamma': ['scale', 'auto', 0.1, 1, 10]  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    }

    # Initialize the grid search, verbose = 2 default but verbose = 0  suppress detailed output
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X, y)

    # Best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated accuracy:", grid_search.best_score_)

    # Evaluate the best model on the test set
    best_svm = grid_search.best_estimator_

    return best_svm
