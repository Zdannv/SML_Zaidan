
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tempfile
import os

df = pd.read_csv("processed_wine.csv")
X=df.drop("quality",axis=1)
y=df["quality"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

params={
    "n_estimators":[100,200],
    "max_depth":[None,10,20],
    "min_samples_split":[2,5]
}

mlflow.set_experiment("Wine Quality Tuning")

with mlflow.start_run():
    grid=GridSearchCV(
        RandomForestClassifier(random_state=42),
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train,y_train)

    best=grid.best_estimator_
    pred=best.predict(X_test)

    acc=accuracy_score(y_test,pred)
    prec=precision_score(y_test,pred,average="weighted")
    rec=recall_score(y_test,pred,average="weighted")
    f1=f1_score(y_test,pred,average="weighted")

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",prec)
    mlflow.log_metric("recall",rec)
    mlflow.log_metric("f1_score",f1)

    with tempfile.TemporaryDirectory() as d:
        report=os.path.join(d,"classification_report.txt")
        matrix=os.path.join(d,"confusion_matrix.csv")

        with open(report,"w") as f:
            f.write(classification_report(y_test,pred))

        pd.DataFrame(confusion_matrix(y_test,pred)).to_csv(matrix,index=False)

        mlflow.log_artifact(report)
        mlflow.log_artifact(matrix)

    mlflow.sklearn.log_model(best,"model")

    print("Best Params:",grid.best_params_)
    print("Accuracy:",acc)
