import lightgbm as lgb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_models(model_save_path, num_folds):
    models = []
    for i in range(1, num_folds + 1):
        model_filename = os.path.join(model_save_path, f'doblez_{i}.txt')
        if os.path.exists(model_filename):
            model = lgb.Booster(model_file=model_filename)
            models.append(model)
        else:
            print(f"Model file {model_filename} not found.")
    return models

def get_combined_feature_importance_df(fold_models, save_path):
    feature_importance_df = pd.DataFrame()

    for i, model in enumerate(fold_models):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = model.feature_name()
        fold_importance_df["Importance"] = model.feature_importance(importance_type="gain")
        fold_importance_df["Fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    if not feature_importance_df.empty:
        avg_feature_importance = feature_importance_df.groupby("Feature").mean().sort_values(by="Importance", ascending=False)
        avg_feature_importance.reset_index().to_csv(save_path, index=False)
        return avg_feature_importance.reset_index()
    else:
        print("No feature importances to display.")
        return pd.DataFrame()

def plot_feature_importances(feature_importances, top_n=35):
    if not feature_importances.empty:
        top_features = feature_importances.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=top_features)
        plt.title('Top 35 Feature Importances')
        plt.tight_layout()
        plt.show()
    else:
        print("No feature importances to plot.")

# Main execution
model_path = r'D:\DS Northeastern\DS 5220 - Supervised Learning\Project Submission\Best Model - 5.3364\5.3364 LightGBM - Best model'
models = load_models(model_path, 5)
combined_feature_importance = get_combined_feature_importance_df(models, '5.3364_feature_importance.csv')
plot_feature_importances(combined_feature_importance)
