import numpy as np
import pandas as pd


def display_results(results):
    print(f"Best parameters are: {results.best_params_}")
    print("\n")
    mean_score = results.cv_results_["mean_test_score"]
    std_score = results.cv_results_["std_test_score"]
    params = results.cv_results_["params"]
    for mean, std, params in zip(mean_score, std_score, params):
        print(f"{round(mean, 3)} + or -{round(std, 3)} for the {params}")


def display_transformations(data_frame, attribute, show_normal_test=True):
    from matplotlib import pyplot as plt
    from scipy.stats.mstats import normaltest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    data = data_frame.copy()

    if (data[f"{attribute}"] <= 0).any():
        pow_tran = PowerTransformer(method="yeo-johnson", standardize=True)
    else:
        pow_tran = PowerTransformer(method="box-cox", standardize=True)

    quan_tran = QuantileTransformer(output_distribution="normal", random_state=0)

    log_scale_tran = make_pipeline(
        FunctionTransformer(np.log1p, validate=False), StandardScaler()
    )

    sqrt_scale_tran = make_pipeline(
        FunctionTransformer(np.sqrt, validate=False), StandardScaler()
    )

    rubust = make_pipeline(RobustScaler())

    data[f"{attribute}_transf_PT"] = make_pipeline(
        SimpleImputer(), pow_tran
    ).fit_transform(data[[attribute]])
    data[f"{attribute}_transf_QT"] = make_pipeline(
        SimpleImputer(), quan_tran
    ).fit_transform(data[[attribute]])
    if ~(data[f"{attribute}"] <= 0).any():
        data[f"{attribute}_transf_log1p"] = make_pipeline(
            SimpleImputer(), log_scale_tran
        ).fit_transform(data[[attribute]])
        data[f"{attribute}_transf_sqrt"] = make_pipeline(
            SimpleImputer(), sqrt_scale_tran
        ).fit_transform(data[[attribute]])
    else:
        data[f"{attribute}_transf_log1p"] = None
        data[f"{attribute}_transf_sqrt"] = None

    data[f"{attribute}_robust_scaler"] = make_pipeline(
        SimpleImputer(), rubust
    ).fit_transform(data[[attribute]])

    # Create two "subplots" and a "figure" using matplotlib
    fig, (
        ax_before,
        ax_after1,
        ax_after2,
        ax_after3,
        ax_after4,
        ax_after5,
    ) = plt.subplots(1, 6, figsize=(15, 5))

    nbins = 15
    # Create a histogram on the "ax_before" subplot
    data[attribute].hist(ax=ax_before, bins=nbins)

    # Apply a log transformation (numpy syntax) to this column
    data[f"{attribute}_transf_PT"].hist(ax=ax_after1, bins=nbins)
    data[f"{attribute}_transf_QT"].hist(ax=ax_after2, bins=nbins)
    data[f"{attribute}_transf_log1p"].hist(ax=ax_after3, bins=nbins)
    data[f"{attribute}_transf_sqrt"].hist(ax=ax_after4, bins=nbins)
    data[f"{attribute}_robust_scaler"].hist(ax=ax_after5, bins=nbins)

    # Formatting of titles etc. for each subplot
    ax_before.set(title="before", ylabel="frequency", xlabel="value")
    ax_after1.set(title="PowerTransformer", ylabel="frequency", xlabel="value")
    ax_after2.set(title="QuantileTransformer", ylabel="frequency", xlabel="value")
    ax_after3.set(title="Log1p", ylabel="frequency", xlabel="value")
    ax_after4.set(title="Sqrt", ylabel="frequency", xlabel="value")
    ax_after5.set(title="Robust", ylabel="frequency", xlabel="value")
    fig.suptitle(f"{attribute} variable transformation")

    if show_normal_test:
        print("Normal test:\n\n")
        print(f'{attribute}: {normaltest(data[f"{attribute}"].values)}')
        print(
            f'{attribute}_transf_PT: {normaltest(data[f"{attribute}_transf_PT"].values)}'
        )
        print(
            f'{attribute}_transf_QT: {normaltest(data[f"{attribute}_transf_QT"].values)}'
        )
        print(
            f'{attribute}_transf_log1p: {normaltest(data[f"{attribute}_transf_log1p"].values)}'
        )
        print(
            f'{attribute}_robust_scaler: {normaltest(data[f"{attribute}_robust_scaler"].values)}'
        )


def report_model_performance(
    X_train,
    X_test,
    y_train,
    y_test,
    fitted_model,
    model_name,
    param_grid="",
    notes="",
    verbose=True,
):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    y_predicted = fitted_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_predicted)
    macro_f1 = f1_score(y_test, y_predicted, average="macro")
    prec = precision_score(y_test, y_predicted, average="macro")
    rec = recall_score(y_test, y_predicted, average="macro")
    accuracy_train = accuracy_score(y_train, fitted_model.predict(X_train))

    if verbose:
        print(f"{model_name}")
        print(f"Accuracy score on test set: {accuracy_test}")
        print(f"F1-score on test set: {macro_f1}")
        print("Train")
        print(classification_report(fitted_model.predict(X_train), y_train))
        print("Test")
        print(classification_report(y_predicted, y_test))

    return {
        "model": model_name,
        "accuracy": accuracy_test,
        "precision": prec,
        "recall": rec,
        "f1": macro_f1,
        "accuracy_train": accuracy_train,
        "notes": notes,
        "params": fitted_model.named_steps["classifier"],
        "param_grid": param_grid,
    }


def fit_models(
    X_train_set,
    y_train_set,
    X_test_set,
    y_test_set,
    preprocessor,
    classifiers_list,
    names_list,
    save_names_list,
    save_folder,
    params,
    cross_val_splits=3,
    verbose=True,
):
    from joblib import dump
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    log = []
    exclude = ["nb", "cb"]

    for i, classifier in enumerate(classifiers_list):

        if not save_names_list[i] in exclude:
            classifier_pipe = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", classifier)]
            )
            CV = GridSearchCV(
                classifier_pipe, params[i], n_jobs=-1, cv=cross_val_splits
            )
            CV.fit(X_train_set, y_train_set)
            dump(CV.best_estimator_, f"{save_folder}{save_names_list[i]}.joblib")
            log.append(
                report_model_performance(
                    X_train_set,
                    X_test_set,
                    y_train_set,
                    y_test_set,
                    CV.best_estimator_,
                    names_list[i],
                    param_grid=params[i],
                    verbose=verbose,
                )
            )

        elif save_names_list[i] == "nb":
            classifier_pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("scaler", MinMaxScaler()),
                    ("classifier", classifier),
                ]
            )
            classifier_pipe.fit(X_train_set, y_train_set)
            dump(classifier_pipe, f"{save_folder}{save_names_list[i]}.joblib")
            log.append(
                report_model_performance(
                    X_train_set,
                    X_test_set,
                    y_train_set,
                    y_test_set,
                    classifier_pipe,
                    "Naive Bayes",
                    param_grid={},
                    verbose=verbose,
                )
            )

    return pd.DataFrame(data=log)


def eval_models(
    X_train_set,
    y_train_set,
    X_test_set,
    y_test_set,
    classifiers_list,
    names_list,
    save_names_list,
    save_folder,
    verbose=True,
):
    from joblib import load

    log = []

    for i, classifier in enumerate(classifiers_list):
        model = load(f"{save_folder}{save_names_list[i]}.joblib")
        log.append(
            report_model_performance(
                X_train_set,
                X_test_set,
                y_train_set,
                y_test_set,
                model,
                names_list[i],
                verbose=verbose,
            )
        )
    return pd.DataFrame(data=log)


def summary_mean_print(df):
    print(
        f"Mean - Acc: {df.accuracy.mean():.3f} -  "
        f"Prec: {df.precision.mean():.3f} - Rec: {df.recall.mean():.3f} - F1: {df.f1.mean():.3f}"
    )


def shap_summary_plot_wrapper(
    X_tr, X_ts, pipeline, features, target_classes, model_name=None
):
    import shap
    from matplotlib import pyplot as plt

    # starting_X = pipeline['preprocessor'].fit_transform(X_tr)
    test_X = pipeline["preprocessor"].transform(X_ts)
    explainer = shap.TreeExplainer(pipeline["classifier"])
    shap_values = explainer.shap_values(test_X)
    fig = plt.figure(figsize=(6, 6))
    shap.summary_plot(
        shap_values,
        test_X,
        feature_names=features,
        plot_type="bar",
        class_names=target_classes,
        title=model_name,
        show=False,
    )
    plt.title(model_name)
    plt.show()
    return shap_values


def get_tuning_curve(
    model,
    parameter_name,
    parameter_list,
    Xtr,
    Xts,
    ytr,
    yts,
    skip_oob=False,
    step_inizialize=False,
):
    from sklearn.metrics import accuracy_score

    oob_list = list()

    # Iterate through all of the possibilities for parameter_name
    for parameter_value in parameter_list:

        if step_inizialize:
            model.named_steps["classifier"] = model.named_steps[
                "classifier"
            ].set_params(**{f"{parameter_name}": parameter_value})
        else:
            model.named_steps["classifier"].set_params(
                **{f"{parameter_name}": parameter_value}
            )

        # Fit the model
        model.fit(Xtr, ytr)

        if skip_oob:
            oob_error = np.NaN
        else:
            # Get the oob error
            oob_error = 1 - model.named_steps["classifier"].oob_score_

        # Get train accuracy
        train_accuracy = accuracy_score(ytr, model.predict(Xtr))

        # Get test accuracy
        test_accuracy = accuracy_score(yts, model.predict(Xts))

        # Store it
        oob_list.append(
            pd.Series(
                {
                    f"{parameter_name}": parameter_value,
                    "oob_err": oob_error,
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy,
                }
            )
        )

    oob_df = pd.DataFrame(data=oob_list)
    return oob_df
