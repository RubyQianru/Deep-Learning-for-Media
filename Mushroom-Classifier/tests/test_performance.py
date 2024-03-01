from sklearn.metrics import classification_report
import numpy as np

def test_model_performance():
    y_pred = np.load('results/y_pred.npy')
    y_test = np.load('data/y_test.npy')
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    # Check recall for class 1 is higher than 90%
    assert report['1']['recall'] > 0.9
    # Check precision for class 0 is higher than 90%
    assert report['0']['precision'] > 0.9

