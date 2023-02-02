from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def rf_cm(Y_test,Y_pred ):
    fig, ax = plt.subplots()
    cm = confusion_matrix(Y_test, Y_pred)
    x_axis_labels = ['TP', 'TN']
    y_axis_labels = ['PP', 'PN']
    sns.heatmap(cm, fmt=".0f", annot=True, linewidths=.5, cmap='PuRd', xticklabels=x_axis_labels)
    ax.set_yticklabels(y_axis_labels, rotation=0, ha='right')
    plt.title('RF-classification CM')
    fig.savefig('../figures/RF_cm.png')
    plt.show()