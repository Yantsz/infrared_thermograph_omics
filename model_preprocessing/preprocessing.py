from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind, levene
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

class HighCorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop)


class TTestFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, label_col='label', alpha=0.05):
        self.label_col = label_col
        self.alpha = alpha
        self.features_ = None

    def fit(self, X, y):
        X = X.copy()
        X[self.label_col] = y
        data_normal = X[X[self.label_col] == 0]
        data_MS = X[X[self.label_col] == 1]
        self.features_ = []
        for col in X.columns:
            if col != self.label_col:
                if levene(data_normal[col], data_MS[col])[1] > 0.05:
                    p_value = ttest_ind(data_normal[col], data_MS[col])[1]
                else:
                    p_value = ttest_ind(data_normal[col], data_MS[col], equal_var=False)[1]

                if p_value < self.alpha:
                    self.features_.append(col)
        return self

    def transform(self, X):
        return X[self.features_]


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, label_col='label'):
        self.label_col = label_col
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.features_ = [col for col in X.columns if col != self.label_col]
        self.scaler.fit(X[self.features_])
        return self

    def transform(self, X):
        transformed = self.scaler.transform(X[self.features_])
        transformed_df = pd.DataFrame(transformed, index=X.index, columns=self.features_)
        return transformed_df


class RLassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, label_col='label'):
        self.label_col = label_col
        self.features_ = None
        pandas2ri.activate()  # 激活转换功能，一次性激活即可

    def fit(self, X, y):
        # 将DataFrame转换为R的数据框
        X = X.copy()
        X[self.label_col] = y
        rdf = pandas2ri.py2rpy(X)
        # 显式传递数据框到R环境中
        ro.globalenv['rdf'] = rdf
        ro.globalenv['label_col'] = self.label_col

        # 定义和执行R代码
        R_Cols = ro.r('''
        library(glmnet)
        library(dplyr)

        df <- rdf
        X <- as.matrix(df %>% select(-label_col))
        Y <- df[[label_col]]

        set.seed(42)
        cv_model <- cv.glmnet(x = X, y = Y, family = 'binomial', alpha = 1, nfolds = 10)

        tiff('./lasso.tiff',width=6*300,height=3*300,units='px',res=300)
        # par(mar=c(5,5,2,1))
        par(mgp=c(1.5, 0.5, 0))
        par(mfrow=c(1,2))
        plot(cv_model,cex=1,cex.axis=0.8,cex.lab=0.8,xlab='Log(λ)')

        plot(cv_model$glmnet.fit,xvar='lambda',label=FALSE,cex=1,cex.axis=0.8,cex.lab=0.8,xlab='Log(λ)')
        dev.off()

        lasso_coefs <- coef(cv_model, s = "lambda.1se")[,1]
        significant_features <- names(lasso_coefs[lasso_coefs != 0])

        significant_features
        ''')
        self.features_ = list(R_Cols)
        # self.features_.append(self.label_col)
        return self

    def transform(self, X):
        if self.features_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        # 确保只选择存在的特征
        available_features = [f for f in self.features_ if f in X.columns]
        return X[available_features]