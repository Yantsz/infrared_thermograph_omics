from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Simhei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('white', {'font.sans-serif': ['Simhei', 'Arial'], 'axes.unicode_minus': False})
from sklearn.metrics import classification_report,roc_auc_score, roc_curve
def plot_roc_curve(y_true, y_scores, label):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f'{label} AUC {auc_score:.2f}')
    plt.legend(loc=4)

def evaluate_model(y_true, y_pred, y_scores, dataset):
    print(f'{dataset} set')
    plot_roc_curve(y_true, y_scores, dataset)
    report = classification_report(y_true, y_pred, output_dict=True)
    print('auc:', roc_auc_score(y_true, y_scores))
    print('accuracy:', report['accuracy'])
    print('sensitivity:', report['1']['recall'])  # Assuming '1' is the positive class
    print('specificity:', report['0']['recall'])  # Assuming '0' is the negative class
    print('\n')