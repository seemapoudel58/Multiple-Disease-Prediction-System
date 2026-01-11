import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.settings import METRICS_DIR, PLOTS_DIR

def plot_metrics(metrics_file=None):
    if metrics_file is None:
        metrics_file = str(METRICS_DIR / 'diabetic_model_metrics.csv')
    '''
        Plot the model performance metrics
    '''
    df = pd.read_csv(metrics_file)  
    metrics = ['Precision', 'Recall', 'F1-Score']
    positive_metrics = ['pos_precision', 'pos_recall', 'pos_f1_score']
    negative_metrics = ['neg_precision', 'neg_recall', 'neg_f1_score']
    overall_metric = 'accuracy'  
    
    positive_data = df[df['Metric'].isin(positive_metrics)]['Test'].values
    negative_data = df[df['Metric'].isin(negative_metrics)]['Test'].values
    overall_data = df[df['Metric'] == overall_metric]['Test'].values   
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))  

    x_pos = np.arange(len(metrics))  # X-axis positions for the bars
    bar_width = 0.3  # Width of the bars

    # Plot bars for positive and negative metrics
    bars1 = plt.bar(x_pos - bar_width/2, positive_data, width=bar_width, color='#b8b8b8', label='Diabetic')
    bars2 = plt.bar(x_pos + bar_width/2, negative_data, width=bar_width, color='#1a80bb', label='Non-Diabetic')

    if len(overall_data) > 0:
        accuracy_bar = plt.bar(len(metrics), overall_data[0], width=bar_width, color='#ffcc00', label='Overall')
        metrics.append('Accuracy')   

    plt.xticks(np.arange(len(metrics)), metrics, fontsize=14)  
    plt.yticks(fontsize=14)  
    plt.ylabel('Value', labelpad=15, fontsize=14)  
    plt.ylim(0, 1)  
    plt.title('Model Performance Metrics (Test Set)', pad=20, size=18, weight='bold')

    # Add text labels on the bars
    for bars in [bars1, bars2, accuracy_bar]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f'{100*height:.2f}%', ha='center', va='bottom', fontsize=14)

    plt.legend(loc='upper left', fontsize=14)  
    plt.grid(True, axis='y', linestyle='--', alpha=0.7) 
    
    plt.tight_layout()  
    plt.savefig(str(PLOTS_DIR / 'metrics_plot.png'), dpi=300, bbox_inches='tight')  # Save plot as PNG
    plt.close()  # Close the plot


def plot_pretty_confusion_matrix(metrics_file=None):
    if metrics_file is None:
        metrics_file = str(METRICS_DIR / 'diabetic_model_metrics.csv')
    '''
        Plot a confusion matrix
    '''
    df = pd.read_csv(metrics_file)
    metrics_dict = dict(zip(df['Metric'], df['Test']))
    confusion_matrix = np.array([
        [metrics_dict['TP'], metrics_dict['FN']],  
        [metrics_dict['FP'], metrics_dict['TN']]
    ])  
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(confusion_matrix,
                     annot=True,
                      cmap='Blues',
                     cbar=False,   
                     square=True,
                     xticklabels=['Diabetic', 'Non-Diabetic'],
                     yticklabels=['Diabetic', 'Non-Diabetic'])  
    plt.title('Confusion Matrix', pad=20, size=26, weight='bold')
    plt.xlabel('Actual Value', labelpad=20, fontsize=20)
    plt.ylabel('Predicted Value', labelpad=20, fontsize=20)   
    for text in ax.texts:
        text.set_color('black')
        text.set_fontsize(24)
        text.set_fontweight('bold')  
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=90) 
    plt.gca().spines[:].set_visible(True)
    plt.gca().spines[:].set_linewidth(2)  
    box_style = dict(facecolor='orange', edgecolor='black', pad=3)
    plt.gca().set_xticklabels(['Diabetic', 'Non-Diabetic'], bbox=box_style)
    plt.gca().set_yticklabels(['Diabetic', 'Non-Diabetic'], bbox=box_style) 
    plt.style.use('dark_background')
    plt.savefig(str(PLOTS_DIR / 'confusion_matrix_custom.png'), dpi=300, bbox_inches='tight')
    plt.close()
    