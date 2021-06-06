
#%%

import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
import numpy as np

dataset = pandas.read_csv('housing.data', delim_whitespace=True, header=None, skip_blank_lines=True)
X = dataset.iloc[:, [0, 12]]
y = dataset.iloc[:, 13]


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = np.array(list(y))

print(X.shape)
print(y.shape)
# seed = 42
# X, y = shuffle(X, y, random_state=seed)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)


#%%

gsc = GridSearchCV(
    estimator=SVR(kernel='rbf'),
    param_grid={
        'C': [0.1, 1, 100, 1000],
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
    },
    cv=10, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

grid_result = gsc.fit(X_train, y_train)
best_params = grid_result.best_params_


print(best_params)
print(grid_result.best_score_)


#%% 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

c_options = [0.1, 1, 100, 1000]
epsilon_options =  [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gamma_options = [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

results = []

for c in c_options: 
    for epsilon in epsilon_options:
        for gamma in gamma_options:
            
            scores = []

            cv = KFold(n_splits=10, shuffle=True)
            for train_index, test_index in cv.split(X_train):
                # print("Train Index: ", train_index, "\n")
                # print("Test Index: ", test_index)

                X_train_part, X_test_part, y_train_part, y_test_part = X[train_index], X[test_index], y[train_index], y[test_index]
                print(X_train_part.shape, X_test_part.shape, y_train_part.shape, y_test_part.shape)

                svn = SVR(kernel='rbf', C=c, gamma=gamma, epsilon=epsilon)
                
                svn.fit(X_train_part, y_train_part)
                
                y_pred_part = svn.predict(X_test_part)
                # scores.append(svn.score(X_test_part, y_test_part))
                scores.append(mean_absolute_error(y_test_part, y_pred_part))

            print(f'c = {c}, epsilon = {epsilon}, gamma = {gamma}, {np.array(scores).mean()}')
            results.append({ 'c': c, 'epsilon': epsilon, 'gamma': gamma, 'score': np.array(scores).mean() })


result_sorted = sorted(results, key=lambda x: x['score'], reverse=False)
print(result_sorted[0])
print(result_sorted[-1])


#%%

import sklearn

print(sklearn.metrics.SCORERS.keys())

# dict_keys([
#   'explained_variance', 
#   'r2', 
#   'max_error', 
#   'neg_median_absolute_error', 
#   'neg_mean_absolute_error', 
#   'neg_mean_absolute_percentage_error', 
#   'neg_mean_squared_error', 
#   'neg_mean_squared_log_error', 
#   'neg_root_mean_squared_error', 
#   'neg_mean_poisson_deviance', 
#   'neg_mean_gamma_deviance', 
#   'accuracy', 
#   'top_k_accuracy', 
#   'roc_auc', 
#   'roc_auc_ovr', 
#   'roc_auc_ovo', 
#   'roc_auc_ovr_weighted', 
#   'roc_auc_ovo_weighted', 
#   'balanced_accuracy', 
#   'average_precision', 
#   'neg_log_loss', 
#   'neg_brier_score', 
#   'adjusted_rand_score', 
#   'rand_score', 
#   'homogeneity_score', 
#   'completeness_score', 
#   'v_measure_score', 
#   'mutual_info_score', 
#   'adjusted_mutual_info_score', 
#   'normalized_mutual_info_score', 
#   'fowlkes_mallows_score', 
#   'precision', 
#   'precision_macro', 
#   'precision_micro', 
#   'precision_samples', 
#   'precision_weighted',  
#   'recall', 
#   'recall_macro', 
#   'recall_micro', 
#   'recall_samples', 
#   'recall_weighted', 
#   'f1', 
#   'f1_macro', 
#   'f1_micro', 
#   'f1_samples', 
#   'f1_weighted', 
#   'jaccard', 
#   'jaccard_macro', 
#   'jaccard_micro', 
#   'jaccard_samples', 
#   'jaccard_weighted'
# ])



#%%

best_svr = SVR(kernel='rbf', 
                C=best_params["C"], 
                epsilon=best_params["epsilon"], 
                gamma=best_params["gamma"],
                # coef0=0.1, 
                # shrinking=True,
                # tol=0.001, 
                # cache_size=200, 
                verbose=False, 
                max_iter=-1)

#%%

print(best_svr)




#%%

clf = best_svr.fit(X_train, y_train)






#%%

y_pred = clf.predict(X_test)

from sklearn.metrics import \
    classification_report, \
    confusion_matrix, \
    accuracy_score, \
    balanced_accuracy_score, \
    mean_squared_error, \
    mean_absolute_error, \
    f1_score, \
    cohen_kappa_score, \
    precision_score, \
    recall_score, \
    r2_score
    
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print(recall_score(y_test, y_pred, average='micro'))
print(r2_score(y_test, y_pred, sample_weight=None))

print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(f1_score(y_test, y_pred, average=None))



#%%

scoring = {
            'abs_error': 'neg_mean_absolute_error',
            'squared_error': 'neg_mean_squared_error'}

scores = cross_validate(best_svr, X, y, cv=10, scoring=scoring, return_train_score=True)
print("MAE :", abs(scores['test_abs_error'].mean()))
print("RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())))




# %%
