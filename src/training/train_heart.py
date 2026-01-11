from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.data_preprocessing import DataPreprocessor
from src.models.logistic_regression import LogisticRegression
from src.utils.model_evaluation import ModelEvaluator
from src.config.settings import RAW_DATA_DIR

def main():
    print("Starting Heart Disease Prediction...")

    # Step 1: Preprocess Data
    print("\nPreprocessing Data...")
    preprocessor = DataPreprocessor(str(RAW_DATA_DIR / "heart.csv"))
    preprocessor.load_data()
    preprocessor.normalize_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()
    preprocessor.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 2: Train Logistic Regression Model
    print("\nTraining Model...")
    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    model.save_model()

    # Step 3: Evaluate Model
    print("\nEvaluating Model...")
    evaluator = ModelEvaluator()
    evaluator.load_model()
    evaluator.load_test_data()

    accuracy, y_pred, conf_matrix, report_df = evaluator.evaluate()

    # Plot and save the confusion matrix
    conf_matrix_path = evaluator.plot_confusion_matrix_heart(conf_matrix)
    print(f"Confusion matrix plot saved to {conf_matrix_path}")

    # Plot and save the metrics
    metrics_path = evaluator.plot_metrics_heart(report_df, accuracy)
    print(f"Metrics plot saved to {metrics_path}")

if __name__ == "__main__":
    main()
