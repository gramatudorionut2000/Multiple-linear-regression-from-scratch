import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)


data=pd.read_csv('insurance.csv')
print(data.isnull().sum())
data=data.drop_duplicates()

sns.histplot(data['charges'],bins=50,color='r', kde=True, stat='density').set_title('Distribution of insurance charges')
plt.show()
OHE_columns = ['sex', 'smoker', 'region']
encoded = pd.get_dummies(data = data, prefix = 'OHE', prefix_sep='_',
               columns = OHE_columns,
               drop_first =False,
              dtype='int8')
encoded=encoded.drop('OHE_no', axis=1)
encoded=encoded.drop('OHE_male', axis=1)
encoded=encoded.rename(columns={'OHE_yes':'smoker', 'OHE_female':'sex'})
columns=['age','sex', 'bmi', 'children', 'smoker', 'OHE_northeast', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest', 'charges']
encoded=encoded.reindex(columns=columns)
for column in ['age', 'bmi']:
    encoded[column] = (encoded[column] - encoded[column].min()) / (encoded[column].max() - encoded[column].min())
encoded['charges']=np.log(encoded['charges'])
sns.histplot(encoded['charges'],bins=50,color='r', kde=True, stat='density').set_title('Distribution of insurance charges-log scale')
plt.show()

corr = encoded[encoded.columns].corr()
sns.heatmap(corr, cmap = 'crest', annot= True)
plt.show()
sns.pairplot(data=encoded, x_vars=columns[:-1], y_vars=columns[-1])
plt.show()

from sklearn.decomposition import PCA

X = encoded.values
pca = PCA(n_components=4)
X_four_d = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = X_four_d[:,0]
y = X_four_d[:,1]
z = X_four_d[:,2]
c = X_four_d[:,3]

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()


X = np.array(encoded.drop('charges', axis=1))  
y = np.array(encoded['charges'])  

dataset = np.column_stack((X, y))
np.random.shuffle(dataset)
metrics=['MSE', 'RMSE', 'MAE', 'NRMSE']
totals=dict()
maximums=dict()
for metric in metrics:
    totals[metric]=0
    maximums[metric]=float('-inf')
    maximums[f'index_{metric}']=float('inf')
k = 10
subsets = np.array_split(dataset, k)

def confidence_interval(data):
    mean, std_error = np.mean(data), np.std(data)/np.sqrt(len(data))
    h = std_error * 1.96
    return np.exp(mean-h), np.exp(mean+h)

for i in range(k):
    test_features = subsets[i][:, :-1]
    test_labels = subsets[i][:, -1]
    training_features = np.vstack([subset for j, subset in enumerate(subsets) if j != i])[:, :-1]
    training_labels = np.vstack([subset for j, subset in enumerate(subsets) if j != i])[:, -1]

    training_features = np.append(np.ones((training_features.shape[0], 1)), training_features, axis=1)
    test_features = np.append(np.ones((test_features.shape[0], 1)), test_features, axis=1)

    theta = np.dot(np.linalg.pinv(training_features), training_labels)
    predictions = np.dot(test_features, theta)
    lower_bound,upper_bound= confidence_interval(predictions)

    test_labels=np.exp(test_labels)
    predictions=np.exp(predictions)
    mse = np.mean((predictions - test_labels) ** 2)
    if maximums['MSE'] < mse:
        maximums['MSE']=mse
        maximums['index_MSE']=i+1
    totals['MSE']=totals['MSE']+mse

    rmse = np.sqrt(mse)
    if maximums['RMSE'] < rmse:
        maximums['RMSE']=rmse
        maximums['index_RMSE']=i+1
    totals['RMSE']=totals['RMSE']+rmse

    absolute_errors = np.abs(test_labels - predictions)
    mae = np.mean(absolute_errors)
    if maximums['MAE'] < mae:
        maximums['MAE']=mae
        maximums['index_MAE']=i+1
    totals['MAE']=totals['MAE']+mae

    nrmse = rmse / (test_labels.max() - test_labels.min())
    if maximums['NRMSE'] < nrmse:
        maximums['NRMSE']=nrmse
        maximums['index_NRMSE']=i+1
    totals['NRMSE']=totals['NRMSE']+nrmse

    
    print(f'-----Fold {i+1}-----')
    print(f"The mean squared error for fold {i+1} is: {mse}")
    print(f"The mean absolute error for fold {i+1} is: {mae}")
    print(f"The root mean squared error for fold {i+1} is: {rmse}")
    print(f"The normalized root mean squared error for fold {i+1} is: {nrmse}")
    print(f"The confidence interval for fold {i+1} is: {lower_bound}-{upper_bound}")
print(f'-----Averages-----')
print(f"The average mean squared error across {k}-folds is: {totals['MSE']/k}")
print(f"The average mean absolute error across {k}-folds is: {totals['MAE']/k}")
print(f"The average root mean squared error across {k}-folds is: {totals['RMSE']/k}")
print(f"The average normalized root mean squared error across {k}-folds is: {totals['NRMSE']/k}")
print(f'-----Maximums-----')
print(f"The maximum mean squared error across {k}-folds is: {maximums['MSE']} in fold {maximums['index_MSE']}")
print(f"The maximum mean absolute error across {k}-folds is: {maximums['MAE']} in fold {maximums['index_MAE']}")
print(f"The maximum root mean squared error across {k}-folds is: {(maximums['RMSE'])} in fold {maximums['index_RMSE']}")
print(f"The maximum normalized root mean squared error across {k}-folds is: {maximums['NRMSE']} in fold {maximums['index_NRMSE']}")
print(f"The mean of the charges variable is: {np.exp(encoded['charges']).mean()}")
print(f"The standard deviation of the charges variable is: {np.exp(encoded['charges']).std()}")