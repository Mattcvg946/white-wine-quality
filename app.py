# Student ID: 13890485
# Student Name: 阮嘉興

from io import BytesIO  # Convert the supplied CSV bytes into a readable stream.
from pathlib import Path  # Resolve the dataset relative to this script.

import altair as alt  # Build interactive distribution and confusion charts.
import numpy as np  # Test that all inputs are finite and group partitions are disjoint.
import pandas as pd  # Load the dataset and label the displayed tables.
import streamlit as st  # Render the web application and cache fitted estimators.
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # Compare three tree-based classifiers.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score  # Compute multiclass performance.
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold  # Keep identical measurements together during model selection and evaluation.
from threadpoolctl import threadpool_limits  # Bound the native threads used during fitting.

SEED = 42  # Make the split and fitted models reproducible.
DATA_PATH = Path(__file__).with_name("winequality-white.csv")  # Use the course-supplied CSV beside this script.
TARGET = "quality"  # Keep the original integer classes.
UNITS = {  # Show the measurement scale beside each interactive input.
    "fixed acidity": "g/L as tartaric acid",  # Fixed acid concentration.
    "volatile acidity": "g/L as acetic acid",  # Volatile acid concentration.
    "citric acid": "g/L",  # Citric acid concentration.
    "residual sugar": "g/L",  # Residual sugar concentration.
    "chlorides": "g/L as sodium chloride",  # Chloride concentration.
    "free sulfur dioxide": "mg/L",  # Free sulfur dioxide concentration.
    "total sulfur dioxide": "mg/L",  # Total sulfur dioxide concentration.
    "density": "g/cm³",  # Density at the dataset's measurement conditions.
    "pH": "",  # pH is unitless.
    "sulphates": "g/L as potassium sulphate",  # Sulphate concentration.
    "alcohol": "% by volume",  # Alcohol percentage by volume.
}  # End the input metadata.
FEATURES = list(UNITS)  # Preserve dataset column order for fitting and prediction.


@st.cache_data(show_spinner=False)  # Parse again only when the supplied bytes change.
def load_data(raw_csv):  # Validate measurements and integer quality labels.
    data = pd.read_csv(BytesIO(raw_csv), sep=";")  # Parse the course file's semicolon delimiter.
    data = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")  # Enforce the expected numeric schema.
    if not np.isfinite(data.to_numpy()).all():  # Detect missing and infinite data before fitting.
        raise ValueError("The dataset contains missing or non-finite values.")  # Explain the data issue.
    if not (data[TARGET].between(0, 10) & data[TARGET].mod(1).eq(0)).all():  # Check the assignment's score definition.
        raise ValueError("Wine quality must be an integer between 0 and 10.")  # Explain the label requirement.
    return data.astype({TARGET: int})  # Store quality as discrete class labels.


@st.cache_resource(show_spinner="Training three classifiers and evaluating the held-out wines…")  # Reuse fits across widget changes and sessions.
def train_models(raw_csv):  # Fit and evaluate every model on a common protocol.
    data = load_data(raw_csv)  # Load all valid observations.
    x, y = data[FEATURES], data[TARGET]  # Separate lab measurements and quality scores.
    labels = np.sort(y.unique())  # Use the same class order throughout evaluation.
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()  # Group rows with identical predictors even when labels differ.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)  # Reserve one approximately stratified group fold.
    train, test = next(outer.split(x, y, groups))  # Select the first predeclared fold without looking at scores.
    assert not np.intersect1d(groups[train], groups[test]).size  # Verify identical measurements cannot leak across partitions.
    x_train, y_train = x.iloc[train], y.iloc[train]  # Restrict tuning and fitting to training rows.
    x_test, y_test = x.iloc[test], y.iloc[test]  # Save the untouched holdout for final evaluation.
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)  # Apply the same grouping policy during tuning.
    def score_macro_f1(model, features, truth):  # Score all observed quality classes equally during selection.
        return f1_score(truth, model.predict(features), average="macro", labels=labels, zero_division=0)  # Evaluate multiclass labels directly.

    specifications = {  # Define small search grids with distinct learning strategies.
        "Random Forest": (  # Aggregate decision trees fit on bootstrap samples.
            RandomForestClassifier(n_estimators=220, random_state=SEED, n_jobs=2),  # Fix ensemble size and limit parallelism.
            {"max_features": ["sqrt", 0.8], "class_weight": [None, "balanced_subsample"]},  # Test feature subsampling and class reweighting.
        ),  # End the random forest specification.
        "Extra Trees": (  # Aggregate trees with randomized split thresholds.
            ExtraTreesClassifier(n_estimators=260, random_state=SEED, n_jobs=2),  # Fix ensemble size and limit parallelism.
            {"min_samples_leaf": [1, 2], "class_weight": [None, "balanced"]},  # Test leaf smoothing and class reweighting.
        ),  # End the extra trees specification.
        "Histogram Gradient Boosting": (  # Fit trees sequentially using binned features.
            HistGradientBoostingClassifier(max_iter=170, early_stopping=False, random_state=SEED),  # Keep training duration fixed across folds.
            {"learning_rate": [0.06, 0.12], "max_leaf_nodes": [15, 31]},  # Tune shrinkage and tree complexity.
        ),  # End the boosting specification.
    }  # End the model definitions.
    results = {}  # Store the model, its tuning result and its held-out predictions together.
    for name, (estimator, grid) in specifications.items():  # Evaluate every model on the same samples.
        search = GridSearchCV(estimator, grid, scoring=score_macro_f1, cv=cv, refit=True, error_score="raise")  # Refit each CV winner using the full training set.
        with threadpool_limits(limits=2, user_api="openmp"):  # Keep boosting responsive on shared hosts.
            search.fit(x_train, y_train, groups=groups[train])  # Supply grouping information to every CV split.
        predicted = search.predict(x_test)  # Generate held-out predictions only after tuning.
        results[name] = {  # Collect everything needed by the dashboard.
            "model": search.best_estimator_,  # Save the refitted prediction model.
            "parameters": search.best_params_,  # Record the cross-validation winner's settings.
            "cv": search.best_score_,  # Record the mean macro F1 from its training folds.
            "predicted": predicted,  # Record predictions in holdout row order.
        }  # End this model's result.
    return results, train, test, labels, len(np.unique(groups))  # Supply evaluation partitions and data diagnostics.


def metric_table(results, truth, labels):  # Assemble comparable performance statistics for all fitted models.
    rows = []  # Accumulate one comparable row per classifier.
    for name, run in results.items():  # Score each classifier's holdout predictions.
        prediction = run["predicted"]  # Look up the predictions made on the common holdout.
        options = {"labels": labels, "average": "macro", "zero_division": 0}  # Include every observed class even if a model never predicts it.
        rows.append({  # Report accuracy and complementary macro and weighted measures.
            "Model": name,  # Identify the model associated with this row.
            "CV macro F1": run["cv"],  # Show the score used for model selection.
            "Accuracy": accuracy_score(truth, prediction),  # Count exact quality matches.
            "Macro precision": precision_score(truth, prediction, **options),  # Average precision equally across labels.
            "Macro recall": recall_score(truth, prediction, **options),  # Average recall equally across labels.
            "Macro F1": f1_score(truth, prediction, **options),  # Average class-level F1 equally across labels.
            "Weighted F1": f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0),  # Weight class F1 by support.
        })  # Append the completed comparison row.
    return pd.DataFrame(rows).sort_values("CV macro F1", ascending=False)  # Rank by training CV, independently of the holdout.


def show_prediction(results, data, recommended):  # Render the lab measurement form and prediction output.
    names = list(results)  # Preserve the declared model choices.
    name = st.selectbox("Classifier", names, index=names.index(recommended), key="predictor")  # Default to the cross-validation winner.
    with st.form("wine_measurements"):  # Submit all measurements together as one sample.
        st.caption("Enter the wine’s lab measurements. Initial values are the training medians.")  # Explain the prefilled values.
        columns = st.columns(3)  # Group compact controls across the available width.
        values = {}  # Collect feature values in the model's expected order.
        for index, (feature, unit) in enumerate(UNITS.items()):  # Add a numeric input for each predictor.
            label = f"{feature.capitalize()} ({unit})" if unit else feature  # Add units and preserve the spelling of pH.
            places = 5 if feature in {"density", "chlorides"} else 3  # Preserve the smaller scale of density and chlorides.
            values[feature] = columns[index % len(columns)].number_input(  # Keep labels and numeric controls aligned.
                label,  # Show the feature name and units.
                min_value=0.0,  # Disallow physically invalid negative values.
                value=round(float(data[feature].median()), places),  # Initialize with the training median.
                step=10.0 ** -places,  # Match the step to the displayed precision.
                format=f"%.{places}f",  # Format floats at the feature's chosen precision.
            )  # Retain the entered value by feature name.
        submitted = st.form_submit_button("Predict quality", type="primary")  # Run inference only when requested.
    if submitted:  # Keep the page idle until a sample is submitted.
        sample = pd.DataFrame([values], columns=FEATURES)  # Match the training column names and order.
        if not np.isfinite(sample.to_numpy()).all():  # Reject invalid numeric inputs.
            st.error("Enter a finite number for each measurement.")  # Return feedback before inference.
            return  # Avoid passing invalid values to the model.
        outside = [feature for feature, value in values.items() if not data[feature].min() <= value <= data[feature].max()]  # Identify inputs beyond the observed training range.
        if outside:  # Flag predictions that require extrapolation.
            st.warning("Outside the observed training range: " + ", ".join(outside))  # Name the affected measurements.
        model = results[name]["model"]  # Retrieve the fitted estimator selected by the user.
        predicted = int(model.predict(sample)[0])  # Obtain the estimated quality score.
        probabilities = pd.DataFrame({"Quality": model.classes_, "Probability": model.predict_proba(sample)[0]})  # Map the estimator's class scores to quality values.
        st.metric("Predicted quality", f"{predicted} / 10")  # Show the main classification result.
        chart = alt.Chart(probabilities).mark_bar(color="#18796b").encode(x=alt.X("Quality:O", title="Quality score"), y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), tooltip=["Quality:O", alt.Tooltip("Probability:Q", format=".1%")])  # Plot the full class probability distribution.
        st.altair_chart(chart, use_container_width=True)  # Display interactive tooltips for each quality score.
        st.caption("These are raw model probability estimates. Their calibration was not evaluated.")  # Describe what the probability bars represent.


def show_evaluation(results, truth, labels, comparison, recommended):  # Render the comparison and selected model diagnostics.
    st.subheader("Classifier comparison")  # Title the shared evaluation table.
    st.caption(f"Highest training CV macro F1 • {recommended}. All other columns use the same held-out wines.")  # Distinguish selection from final evaluation.
    st.dataframe(comparison.style.format(precision=3), hide_index=True, width="stretch")  # Present all requested metrics side by side.
    name = st.selectbox("Inspect classifier", list(results), index=list(results).index(recommended), key="evaluation")  # Let users switch diagnostic views.
    run = results[name]  # Retrieve the selected model's evaluation outputs.
    predicted = run["predicted"]  # Keep diagnostics tied to the held-out predictions.
    matrix = pd.DataFrame(confusion_matrix(truth, predicted, labels=labels), index=pd.Index(labels, name="Actual"), columns=pd.Index(labels, name="Predicted"))  # Label actual and predicted axes explicitly.
    cells = matrix.stack().rename("Wines").reset_index()  # Convert the matrix to chart-ready rows.
    base = alt.Chart(cells).encode(x=alt.X("Predicted:O", title="Predicted quality"), y=alt.Y("Actual:O", title="Actual quality"), tooltip=["Actual:O", "Predicted:O", "Wines:Q"])  # Share chart axes and tooltips between cells and labels.
    heatmap = base.mark_rect().encode(color=alt.Color("Wines:Q", scale=alt.Scale(scheme="teals")))  # Color each cell by its wine count.
    counts = base.mark_text().encode(text="Wines:Q", color=alt.condition(alt.datum.Wines > matrix.to_numpy().max() / 2, alt.value("white"), alt.value("#163631")))  # Keep counts readable on dark and light cells.
    st.altair_chart((heatmap + counts).properties(title="Holdout confusion matrix", height=320), use_container_width=True)  # Show exact counts and error patterns.
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)  # Calculate per-class diagnostics in score order, with undefined metrics set to zero.
    report = pd.DataFrame({"Quality": labels, "Precision": precision, "Recall": recall, "F1": f1, "Holdout wines": support})  # Include support so users can interpret rare-score results.
    st.dataframe(report.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), hide_index=True, width="stretch")  # Display per-class results.
    st.caption("Macro metrics weight each score equally. Weighted F1 weights scores by holdout frequency. Rare scores have limited support, so their metrics can vary substantially across splits.")  # Explain the class-imbalance trade-off.
    st.write("Selected parameters", run["parameters"])  # Show which settings won the model's training search.


def show_dataset(data, train, test, labels, group_count):  # Show how the data and evaluation split are composed.
    overview = st.columns(3)  # Place the main data sizes side by side.
    overview[0].metric("Wine records", f"{len(data):,}")  # Count all supplied records, including repeated measurements.
    overview[1].metric("Training records", f"{len(train):,}")  # Show the actual grouped training split size.
    overview[2].metric("Holdout records", f"{len(test):,}")  # Show the actual grouped holdout size.
    st.caption("Quality scores present in the supplied data: " + ", ".join(map(str, labels)))  # List the classes the classifiers can learn.
    train_counts = data.iloc[train][TARGET].value_counts().reindex(labels, fill_value=0)  # Count all observed labels in training.
    test_counts = data.iloc[test][TARGET].value_counts().reindex(labels, fill_value=0)  # Count all observed labels in the holdout.
    split = pd.DataFrame({"Quality": labels, "Training": train_counts.to_numpy(), "Holdout": test_counts.to_numpy()})  # Align counts by quality.
    plot = split.melt(id_vars="Quality", var_name="Partition", value_name="Wines")  # Reshape into one row per label and split.
    chart = alt.Chart(plot).mark_bar().encode(x=alt.X("Quality:O", title="Quality score"), y=alt.Y("Wines:Q", title="Wine records"), color=alt.Color("Partition:N", scale=alt.Scale(domain=["Training", "Holdout"], range=["#18796b", "#d1a141"])), tooltip=["Quality:O", "Partition:N", "Wines:Q"])  # Show imbalance and each partition's contribution to the class counts.
    st.altair_chart(chart.properties(title="Quality distribution"), use_container_width=True)  # Render the class distribution plot.
    st.dataframe(split, hide_index=True, width="stretch")  # Provide exact counts including rare classes.
    st.caption(f"{group_count:,} unique measurement profiles. Identical feature rows stay in the same partition during the holdout split and cross-validation.")  # Describe the duplicate-handling rule.
    with st.expander("Evaluation protocol"):  # Make the experiment reproducible from the interface.
        st.write("The first fold of a five-fold stratified group split is held out. Each classifier’s parameter grid is ranked by four-fold stratified group cross-validation macro F1 on the training partition. Each selected configuration is refit on the entire training partition and then evaluated on the holdout. The fixed random seed is 42.")  # Explain split, model selection and final evaluation in order.
    with st.expander("Preview supplied records"):  # Provide a compact view of the original dataset.
        st.dataframe(data.head(20), hide_index=True, width="stretch")  # Show the first twenty records and all columns.


def main():  # Load the data and render each part of the application.
    st.set_page_config(page_title="White Wine Quality", page_icon="🍇", layout="wide")  # Set browser metadata and a wide dashboard.
    st.title("White Wine Quality")  # Identify the application and its prediction task.
    st.caption("Machine Learning Assignment Two · 13890485 · 阮嘉興")  # Display assignment attribution.
    try:  # Report input-file problems within the page.
        raw_csv = DATA_PATH.read_bytes()  # Pass the actual bytes to the caches so dataset edits invalidate results.
        data = load_data(raw_csv)  # Parse and validate the supplied file.
    except (OSError, ValueError, KeyError) as error:  # Handle missing files, invalid data and unexpected headers.
        st.error(f"Could not load winequality-white.csv. Keep the course CSV beside app.py. Details: {error}")  # Explain how to resolve the input problem.
        st.stop()  # Avoid constructing the dashboard when its input is unavailable.
    results, train, test, labels, group_count = train_models(raw_csv)  # Run or retrieve the cached experiment.
    truth = data.iloc[test][TARGET]  # Extract the holdout ground truth in prediction order.
    comparison = metric_table(results, truth, labels)  # Build the comparison table once per page render.
    recommended = comparison.iloc[0]["Model"]  # Use training CV macro F1 to choose the default model.
    predict_tab, evaluation_tab, dataset_tab = st.tabs(["Predict quality", "Model evaluation", "Dataset"])  # Provide direct navigation among the three tasks.
    with predict_tab:  # Render the prediction form.
        show_prediction(results, data.iloc[train], recommended)  # Use the training partition for defaults and range checks.
    with evaluation_tab:  # Render scores and diagnostics.
        show_evaluation(results, truth, labels, comparison, recommended)  # Compare and inspect holdout performance.
    with dataset_tab:  # Render the data exploration view.
        show_dataset(data, train, test, labels, group_count)  # Provide sample sizes, class support and the protocol.


if __name__ == "__main__":  # Enter the app when invoked by Streamlit.
    main()  # Render the complete dashboard.
