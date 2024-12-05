import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of all uploaded file paths
file_paths = [
    "/Users/slaya/improved-knn/KNNOutputNCAProperPipelined.csv",
    "/Users/slaya/improved-knn/KNNOutputMinkowski.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatUnscaledAccuracy.csv",
    "/Users/slaya/improved-knn/KNNOutputHassanatScaled.csv",
]

# Define column names
columns = ["Dataset", "K", "Accuracy", "Precision", "Recall"]

# List of selected Minkowski p-values
selected_p_values = ["1.0", "1.7411011265922482", "2.0", "2.2973967099940698", "4.0", "8.0"]

def load_and_prepare_data(file_path):
    """Load and clean data from CSV files."""
    try:
        if "Minkowski" in file_path:
            # Special handling for Minkowski to filter by selected p-values
            with open(file_path, "r") as file:
                data = []
                current_p = None
                for line in file:
                    if "Distance Metric,minkowski" in line:
                        current_p = line.split("p=")[-1].strip()
                    elif line.strip() and not line.startswith("Distance Metric") and current_p in selected_p_values:
                        parts = line.strip().split(",")
                        if len(parts) == 5:
                            parts.append(f"Minkowski (p={current_p})")
                            data.append(parts)
            df = pd.DataFrame(data, columns=columns + ["Algorithm"])
        else:
            # Standard processing for other files
            df = pd.read_csv(file_path, header=None, skiprows=1, on_bad_lines="skip")
            df = df[df.columns[:len(columns)]]
            df.columns = columns
            df["Algorithm"] = file_path.split("/")[-1].replace(".csv", "")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_best_k_metrics(data):
    """Extract the best metrics for each algorithm and dataset."""
    try:
        data["Accuracy"] = pd.to_numeric(data["Accuracy"], errors="coerce")
        data.dropna(subset=["Accuracy"], inplace=True)
        best_metrics = data.loc[data.groupby(["Algorithm", "Dataset"])["Accuracy"].idxmax()]
        return best_metrics
    except Exception as e:
        print(f"Error extracting best k metrics: {e}")
        return pd.DataFrame()

def plot_comparative_graph(data):
    """Generate a comparative graph of Accuracy for the best k values."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="Dataset", y="Accuracy", hue="Algorithm", ci=None, dodge=True)
    plt.title("Comparison of Accuracy for Best K Values Across Algorithms")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_files(file_paths):
    """Analyze and compare the best k metrics from all files."""
    all_data = []
    for path in file_paths:
        df = load_and_prepare_data(path)
        if df is not None:
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        best_k_metrics = get_best_k_metrics(combined_data)
        print("Best Metrics for Each Algorithm and Dataset:")
        print(best_k_metrics)
        plot_comparative_graph(best_k_metrics)

if __name__ == "__main__":
    analyze_files(file_paths)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # List of all uploaded file paths
# file_paths = [
#     "/Users/slaya/improved-knn/KNNOutputNCAProperPipelined.csv",
#     "/Users/slaya/improved-knn/KNNOutputMinkowski.csv",
#     "/Users/slaya/improved-knn/KNNOutputHassanatUnscaledAccuracy.csv",
#     "/Users/slaya/improved-knn/KNNOutputHassanatScaled.csv",
# ]

# # Define column names
# columns = ["Dataset", "K", "Accuracy", "Precision", "Recall"]

# def load_and_prepare_data(file_path):
#     """Load and clean data from CSV files."""
#     try:
#         df = pd.read_csv(file_path, header=None, skiprows=1, on_bad_lines="skip")
#         df = df[df.columns[:len(columns)]]
#         df.columns = columns
#         df["Algorithm"] = file_path.split("/")[-1].replace(".csv", "")
#         return df
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

# def get_best_k_metrics(data):
#     """Extract the best metrics for each algorithm and dataset."""
#     try:
#         data["Accuracy"] = pd.to_numeric(data["Accuracy"], errors="coerce")
#         data.dropna(subset=["Accuracy"], inplace=True)
#         best_metrics = data.loc[data.groupby(["Algorithm", "Dataset"])["Accuracy"].idxmax()]
#         return best_metrics
#     except Exception as e:
#         print(f"Error extracting best k metrics: {e}")
#         return pd.DataFrame()

# def plot_comparative_graph(data):
#     """Generate a comparative graph of Accuracy for the best k values."""
#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=data, x="Dataset", y="Accuracy", hue="Algorithm", ci=None)
#     plt.title("Comparison of Accuracy for Best K Values Across Algorithms")
#     plt.xlabel("Dataset")
#     plt.ylabel("Accuracy")
#     plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# def analyze_files(file_paths):
#     """Analyze and compare the best k metrics from all files."""
#     all_data = []
#     for path in file_paths:
#         df = load_and_prepare_data(path)
#         if df is not None:
#             all_data.append(df)

#     if all_data:
#         combined_data = pd.concat(all_data, ignore_index=True)
#         best_k_metrics = get_best_k_metrics(combined_data)
#         print("Best Metrics for Each Algorithm and Dataset:")
#         print(best_k_metrics)
#         plot_comparative_graph(best_k_metrics)

# if __name__ == "__main__":
#     analyze_files(file_paths)