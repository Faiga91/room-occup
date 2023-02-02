from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def rf_classification(df):
    X, Y = df.iloc[:,:-1], df.iloc[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    Y_pred, rf_score, predit_proba = rf.predict(X_test), rf.score(X_test, Y_test), rf.predict_proba(X_test)
    print('Accuracy of Random Forest Classifier on test set: {:.6f}%'.format(rf_score*100))
    return  Y_test, Y_pred, rf_score, predit_proba
