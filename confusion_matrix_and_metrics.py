import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def basic_metrics(y_true, y_pred, class_names, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm[:,::-1][::-1]
    np.set_printoptions(precision=4)
    
    title='Confusion matrix'
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    print(classification_report(y_true, y_pred, target_names=class_names[::-1],digits=4))
    print("Close figure to terminate.")
    plt.show()

def main():
    # Both Binary and multi classification works here
    class_names = np.array(["Two","One", "Zero"]) #Name of classes in descending order
    y_true = [1,1,0,1,0,1,1,2] #True value of class labels 
    y_pred = [1,0,1,1,0,1,0,2] #Predicted value of class labels
    basic_metrics(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()
    