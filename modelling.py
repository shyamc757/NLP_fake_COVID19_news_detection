from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

def return_classification_accuracy(X,y):
    model = LogisticRegressionCV(cv = 5, random_state = 1729, max_iter = 1000, n_jobs = -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 1729)
    LR_model = model.fit(X_train, y_train)
    accuracy = LR_model.score(X_test,y_test)
    return accuracy