# AUTHOR: Dillon Carlos (kaggle.com/djcarlos)
# CREDIT FOR NB-SVM CLASS BASIS -- AlexSánchez (kaggle.com/kniren)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pandas as pd
import numpy as np
import re
import string


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C, dual=False, n_jobs=-1):
        self.Cs = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p+1) / ((y == y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegressionCV(Cs=self.Cs, dual=self.dual, n_jobs=self.n_jobs, solver='liblinear',
                                         cv=10, scoring='neg_log_loss').fit(x_nb, y)
        return self


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)


# The following code block was adopted from Jeremy Howard (kaggle.com/jhoward)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)


x_train = vec.fit_transform(train['comment_text'])
x_test = vec.transform(test['comment_text'])

NbSvm = NbSvmClassifier(C=list(np.arange(0.5, 6.5, 0.5)), dual=True, n_jobs=-1)
preds = np.zeros((len(test), len(label_cols)))


for i, j in enumerate(label_cols):
    print('fit', j)
    NbSvm.fit(x_train, train[j])
    preds[:, i] = NbSvm.predict_proba(x_test)[:, 1]


submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)
