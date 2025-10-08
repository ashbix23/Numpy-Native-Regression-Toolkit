from linear_scratch.plotting import plot_training_history, bar_feature_importance, residuals_plot

def plot_model_insights(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    plot_training_history(model.cost_history, model.val_cost_history)
    bar_feature_importance(model.weight, feature_names)
    residuals_plot(y_test, y_pred)

