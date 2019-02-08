import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
#данные о производстве вина в Австралии за несколько лет
dataset = pd.read_csv("wine_Australia.tsv", sep = '\t') #параметр sep задает разделитель данных. Формат tsv - разделителем является табуляция
print(dataset.head())

#берем из столбца spark (шампанское) все значения. Очевидно, само название нам не нужно
time_series = dataset['spark'].values
#смотрим на временной ряд. Периоды есть? Тренд есть?
plt.plot(time_series, color = 'blue')

#задаем вектора параметров (см. лекции). Поиграйте с этими параметрами, чтобы получить хороший результат (тут специально все плохо)
pdq1 = (1, 1, 1)
PDQS1 = (1, 1, 1, 12)
pdq2 = (1, 1, 3)
PDQS2 = (1, 1, 1, 12)
model1 = SARIMAX(time_series, order = pdq1, seasonal_order = PDQS1)
model2 = SARIMAX(time_series, order = pdq2, seasonal_order = PDQS2)

#тренируем модель и дела10м предсказание
model_fit1 = model1.fit()
predicted1 = model_fit1.predict(start=1, end=len(time_series) + 8) 
model_fit2 = model2.fit()
predicted2 = model_fit2.predict(start=1, end=len(time_series) + 8)
#смотрим на наше предсказание
plt.plot(predicted1, color = 'red')
plt.plot(predicted2, color = 'green')
plt.show()

