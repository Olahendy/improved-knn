import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of all uploaded file paths
file_paths = [
    "/Users/slaya/improved-knn/KNNOutputNCAProperPipelined.csv",
    "/Users/slaya/improved-knn/KNNOutputNCAExpandedK.csv",
    "/Users/slaya/improved-knn/KNNOutputMinkowski.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatUnscaledRecall.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatUnscaledPrecision.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatUnscaledAccuracy.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatScaled.csv",
    "/Users/slaya/improved-knn/KNNOutputEuclideanComponentsAnalysis.csv"
]

# Define column mapping
columns = ["Dataset", "K", "Accuracy", "Precision", "Recall"]

def load_and_prepare_data(file_path):
    """Load and clean data from CSV files."""
    try:
        # Load the CSV with `on_bad_lines='skip'` for pandas >=1.3.0
        df = pd.read_csv(file_path, header=None, skiprows=1, on_bad_lines="skip")
        # Restrict to relevant columns
        df = df[df.columns[:len(columns)]]
        # Rename columns
        df.columns = columns
        # Add source file identifier
        df["Algorithm"] = file_path.split("/")[-1].replace(".csv", "")
        # Drop rows with NaN in Accuracy
        df.dropna(subset=["Accuracy"], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def aggregate_best_metrics(data):
    """Aggregate the best-performing metrics for each dataset and algorithm."""
    try:
        # Ensure no NaN values in Accuracy
        data["Accuracy"] = pd.to_numeric(data["Accuracy"], errors="coerce")
        data.dropna(subset=["Accuracy"], inplace=True)
        # Find the best metrics
        return data.loc[data.groupby(["Algorithm", "Dataset"])["Accuracy"].idxmax()]
    except Exception as e:
        print(f"Error in aggregation: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def plot_comparative_metrics(data, metric):
    """Generate a comparative bar plot for a specific metric."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="Dataset", y=metric, hue="Algorithm", ci=None)
    plt.title(f"Comparison of {metric.capitalize()} Across Algorithms")
    plt.xlabel("Dataset")
    plt.ylabel(metric.capitalize())
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_files(file_paths):
    """Load, process, and compare results from all files."""
    all_data = []
    for path in file_paths:
        df = load_and_prepare_data(path)
        if df is not None:
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Aggregate the best metrics
        best_metrics = aggregate_best_metrics(combined_data)
        print("Best Metrics for Each Algorithm and Dataset:")
        print(best_metrics)
        
        # Plot metrics for comparison
        for metric in ["Accuracy", "Precision", "Recall"]:
            plot_comparative_metrics(best_metrics, metric)

if __name__ == "__main__":
    analyze_files(file_paths)
