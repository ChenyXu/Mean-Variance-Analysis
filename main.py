import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

# get the components of SP500 and drop all the stocks without enough data
data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
df = yf.download(data, start='2010-01-01', interval='1mo')['Adj Close'].dropna(axis=1)
df = np.log(df / df.shift(1)).dropna()


# function that calculates the annualized portfolio standard deviation
def get_std(w, df):
    std = (w.dot(df.cov()).dot(w)) ** 0.5
    return std * 12 ** 0.5


# run optimization: find the minimal standard deviation under certain expected return
res_list = []
steps = np.arange(0.01, 0.3, 0.01)
w = np.ones(len(df.columns)) / np.ones(len(df.columns)).sum()  # the weight list

for i in steps:
    cons = [{'type': 'eq', 'fun': lambda x: df.mean().dot(x) * 12 - i},  # return = target return
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # sum of individual weights = 1
    res = minimize(get_std, w, df, constraints=cons, bounds=Bounds(0, 1), method='SLSQP')
    res_list.append(res['fun'])
    print(res)
    print(i, 'done')


# create plot
x = res_list
y = steps
optimal_x = 0
spot = dict(zip(x, y))
df = pd.DataFrame({'std': x, 'return': y})


def slope(x):
    k = (spot[x] - 0.02) / x
    return k


for i in x:
    if slope(i) == np.max([slope(o) for o in x]):
        optimal_x = i

plt.scatter(x=df['std'], y=df['return'])
plt.axvline(x=min(x), color='r', linestyle='dashed', label='mean variance')
plt.plot(0, 0.01, marker='o', markersize=10, color='y')
plt.annotate('mean-variance portfolio {}'.format((round(min(x), 3), spot[min(x)])), xy=(min(x), spot[min(x)]),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('optimal portfolio {}'.format((round(optimal_x, 3), spot[optimal_x])), xy=(optimal_x, spot[optimal_x]),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('risk free rate 1%', xy=(0, 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('mean-variance analysis of SP500 components')
plt.ylim(bottom=-0.1, top=0.4)
plt.xlim(left=0, right=0.3)
plt.xlabel('std')
plt.ylabel('return')
plt.legend()
plt.show()
