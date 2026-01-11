import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.decision_tree import DecisionTree
from src.utils.visualization import plot_metrics, plot_pretty_confusion_matrix
from src.config.settings import RAW_DATA_DIR, SAVED_MODELS_DIR, METRICS_DIR

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    ''' 
        Split the data into train and test sets while preserving the class distribution
    ''' 
    if random_state is not None:
        np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)
    classes = np.unique(y)
    class_indices = [np.where(y == c)[0] for c in classes]
    train_indices = []
    test_indices = []
    for indices in class_indices:
        n_samples = len(indices)
        n_test = int(test_size * n_samples)     
        shuffled_indices = indices[np.random.permutation(n_samples)]      
        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])   
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)   
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]  
    return X_train, X_test, y_train, y_test


def calculate_metrics_total(y_true, y_pred):
    ''' 
        Calculate the confusion matrix metrics for the given true and predicted labels
    '''  
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    total = len(y_true)
    accuracy = (TP + TN) / total 
    pos_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    pos_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
    pos_support = np.sum(y_true == 1)  
    neg_precision = TN / (TN + FN) if (TN + FN) > 0 else 0
    neg_recall = TN / (TN + FP) if (TN + FP) > 0 else 0
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
    neg_support = np.sum(y_true == 0)  
    return {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN), 
            'accuracy': float(accuracy),  
            'pos_precision': float(pos_precision),
            'pos_recall': float(pos_recall),
            'pos_f1_score': float(pos_f1),
            'pos_support': int(pos_support),       
            'neg_precision': float(neg_precision),
            'neg_recall': float(neg_recall),
            'neg_f1_score': float(neg_f1),
            'neg_support': int(neg_support)
    }

def preprocess_diabetes_data():
    ''' 
        Preprocess the diabetes dataset and return the train and test data
    '''
    diabetes_dataset = pd.read_csv(str(RAW_DATA_DIR / 'diabetes.csv'))
    X = diabetes_dataset.drop(columns='Outcome').to_numpy()
    Y = diabetes_dataset['Outcome'].to_numpy()
    X_train, X_test, Y_train, Y_test = stratified_train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    return X_train, X_test, Y_train, Y_test


def save_model(classifier):
    '''
        Save the trained model to a file
    '''
    filename = str(SAVED_MODELS_DIR / 'diabetes_model_decision_tree.sav')
    pickle.dump(classifier, open(filename, 'wb'))


def get_classification_metrics(Y_train, X_train_prediction, Y_test, X_test_prediction):
    '''
        Calculate the classification metrics for the model
    '''
    train_metrics = calculate_metrics_total(Y_train, X_train_prediction)
    test_metrics = calculate_metrics_total(Y_test, X_test_prediction)
    metrics_df = pd.DataFrame({
        'Metric': list(train_metrics.keys()),
        'Training': list(train_metrics.values()),
        'Test': list(test_metrics.values())
    })  
    metrics_df.to_csv(str(METRICS_DIR / 'diabetic_model_metrics.csv'), index=False)


def plot_model_performance():
    '''
        Plot the model performance metrics
    '''
    metrics_file = str(METRICS_DIR / 'diabetic_model_metrics.csv')
    plot_metrics(metrics_file)
    plot_pretty_confusion_matrix(metrics_file)


def train_diabetic_model():
    '''
        Train the diabetic model and save the model
    '''
    X_train, X_test, Y_train, Y_test = preprocess_diabetes_data()
    classifier = DecisionTree(max_depth=3)
    classifier.fit(X_train, Y_train) 
    X_train_prediction = classifier.predict(X_train)
    X_test_prediction = classifier.predict(X_test)  
    save_model(classifier)
    get_classification_metrics(Y_train, X_train_prediction, Y_test, X_test_prediction)
    plot_model_performance()


if __name__ == '__main__':
    train_diabetic_model()
    print('Training done')

