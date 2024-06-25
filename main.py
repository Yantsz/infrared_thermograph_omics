from feature_extraction.extractor import *
from model_preprocessing.preprocessing import *
from model_evaluation.ml_metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline
from radiomics import featureextractor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 6
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_palette(['#4DBBD5B2','#998385B2','#FFBF86B2','#E64B35B2'])

#1 Set the pyradiomics feature extraction parameters
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.loadParams('./radiomics.yaml')

#2 Extracting Radiomic Features from Facial and Palmar Images.
input_dir_palm = 'd:/infrared_data'
input_dir_face = 'd:/infrared_data'

df_palm = generate_result_xlsx(extractor, input_dir_palm, 'palm')
df_face = generate_result_xlsx(extractor, input_dir_face, 'face')

df_palm2=df_palm.copy()
df_face2=df_face.copy()
df_face2.columns=[x+'_face' if x!='label' else x for x in df_face2.columns]
df_palm2.columns=[y+'_palm' if y != 'label' else y for y in df_palm2.columns ]
df_merge=pd.merge(df_palm2.iloc[:,:-1],df_face2,left_index=True,right_index=True,how='left')

#3 Constructing a sklearn machine learning pipeline
pipeline = Pipeline([
    ('high_corr_filter', HighCorrelationFilter(threshold=0.99)),  # You can adjust the threshold
    ('t_test_selector', TTestFeatureSelector(label_col='label', alpha=0.05)),
    ('scaler', CustomStandardScaler(label_col='label')),
    ('Rlasso_selector', RLassoFeatureSelector(label_col='label')),
    ('classifier', LogisticRegression(C=0.1, solver='lbfgs'))
])

X_train, X_test, y_train, y_test = train_test_split(df_merge.iloc[:,:-1],df_merge.iloc[:,-1], shuffle=True, random_state=1496, test_size=0.3,stratify=df_merge.iloc[:,-1])
pipeline.fit(X_train,y_train)
#4 Evaluating on training and testing set
y_pred_train = pipeline.predict(X_train)
y_scores_train = pipeline.predict_proba(X_train)[:, 1]
evaluate_model(y_train, y_pred_train, y_scores_train, 'Training')

y_pred_test = pipeline.predict(X_test)
y_scores_test = pipeline.predict_proba(X_test)[:, 1]
evaluate_model(y_test, y_pred_test, y_scores_test, 'Testing')

evaluate_model(y_test, y_pred_test, y_scores_test, 'Testing')


pipeline2 = Pipeline([
    ('high_corr_filter', HighCorrelationFilter(threshold=0.99)),  # You can adjust the threshold
    ('t_test_selector', TTestFeatureSelector(label_col='label', alpha=0.05)),
    ('scaler', CustomStandardScaler(label_col='label')),
    ('Rlasso_selector', RLassoFeatureSelector(label_col='label'))])

#5 Visualization and Comparison of Logistic Regression Model with Imaging Biomarker Features and Average Features

pipeline2.fit(X_train,y_train)
X,y=pipeline2.transform(X_train),y_train
model = pipeline['classifier']


coefficients = model.coef_[0]

feature_names = model.feature_names_in_
coefs = coefficients
indices = np.argsort(coefs)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.16, 2.7),width_ratios=[9,8])

bars = ax1.barh(range(X.shape[1]), coefs[indices], color='#1c61b6')
ax1.set_yticks(range(X.shape[1]), [feature_names[i] for i in indices],fontsize=6)

for bar in bars:
    width = bar.get_width()
    label_x_pos = width+0.01 if width > 0 else width - 0.16
    ax1.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center',fontsize=6)

ax1.set_xlabel('Coefficient Value')
ax1.set_ylabel('Features')
ax1.set_xlim(-0.8,0.8)
ax1.set_yticklabels(labels=ax1.get_yticklabels(), horizontalalignment='right', rotation=0)

for col in ['original_firstorder_Mean_face', 'original_firstorder_Mean_palm','wavelet-LH_firstorder_Mean_face','wavelet-LH_glcm_InverseVariance_palm']:
    if col in ['wavelet-LH_firstorder_Mean_face']:
        y_true_r=1-df_merge['label'].astype(int)
        fpr,tpr,thr=roc_curve(y_true_r,df_merge[col])
        ax2.plot(fpr,tpr,label=col+' AUC:'+str(round(auc(fpr,tpr),2)),linewidth=1)
    else:
        fpr,tpr,thr=roc_curve(df_merge['label'],df_merge[col])
        ax2.plot(fpr,tpr,label=col+' AUC:'+str(round(auc(fpr,tpr),2)),linewidth=1)
ax2.plot([0,1],[0,1],linestyle=':',linewidth=1)
ax2.legend(loc=4,fontsize=5)
ax2.set_xlabel('1-Specificity')
ax2.set_ylabel('Sensitivity')
fig.savefig('./logistic_regression_forest2.tiff',bbox_inches='tight',dpi=300)

#6 Model Evaluation Using R Language in Python (ROC Curve, Calibration Curve, Decision Curve)
result_train=pd.DataFrame(data={'y_train':y_train,'y_pred_train':y_scores_train})
result_test=pd.DataFrame(data={'y_test':y_test,'y_pred_test':y_scores_test})
result_train.to_excel('./result_train.xlsx',index=False)
result_test.to_excel('./result_test.xlsx',index=False)

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()

ro.r(
    '''
library(readxl)
library(tidyverse)
library(pROC)

source('./model_evaluation/dca.R')
r_train <- read_excel('./result_train.xlsx')
r_test <- read_excel('./result_test.xlsx')

roc_train<-roc(r_train$y_train,r_train$y_pred_train)
roc_test<-roc(r_test$y_test,r_test$y_pred_test)

auc_train_value <- auc(roc_train)
ci_value_train <- str_c(sprintf("%.2f",round(ci(roc_train)[1],2)),'—',round(ci(roc_train)[3],2))
(legend_train <- str_c('Training    AUC:',sprintf("%.2f",round(auc_train_value,2)),';  ','95%CI:',ci_value_train))

(auc_test_value <- auc(roc_test))
(ci_value_test <- str_c(sprintf("%.2f",round(ci(roc_test)[1],2)),'—',round(ci(roc_test)[3],2)))
(legend_test <-  str_c('Validation AUC:',sprintf("%.2f",round(auc_test_value,2)),';  ','95%CI:',ci_value_test))

tiff('./evaluation.tiff',width=7.16*300,height=2.5*300,units='px',res=300)
par(mfrow=c(1,3))
par(meg=c(1.5,0.5,0))
par(mar=c(4.5,4,2,1))
plot(1-roc_train$specificities,
     roc_train$sensitivities,
     type="l",
     col='#4DBBD5B2',
     lty=1,
     lwd=1.5,
     # twd=2,
     xlab='1-Specificities',
     ylab='Sensitivities',
     xlim=c(0,1),
     ylim=c(0,1),
     xaxs='i',
     yaxs='i',
     cex=1.5,
     cex.lab=1,
     cex.axis=1)
lines(1-roc_test$specificities,
      roc_test$sensitivities,
      type="l",
      col='#E64B35B2',
      lty=1,
      lwd=1.5,
      # twd=2,
      xlab='1-Specificities',
      ylab='Sensitivities',
      xlim=c(0,1),
      ylim=c(0,1),
      xaxs='i',
      yaxs='i',
      cex=1.5,
      cex.lab=1,
      cex.axis=1)

abline(0, 1, lty=2, col="#1c61b6")
legend("bottomright",
       legend=c(legend_train,legend_test),
       lty=c(1,1),
       lwd=c(2,2),
       col=c('#4DBBD5B2','#E64B35B2'),
       bty='n',
       cex=0.6)


library(rms)
calib_data <- val.prob(r_test$y_pred_test, as.numeric(r_test$y_test),stat=F,cex=0.6)

r_test <- as.data.frame(r_test)
dca(data=r_test, outcome = 'y_test',predictors=c('y_pred_test'))
par(mar=c(5,5,2,1))
par(mfrow=c(1,2))

dev.off()
    '''
)