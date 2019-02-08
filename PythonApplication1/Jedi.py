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
pdq = (2, 1, 1)
PDQS = (1, 1, 1, 10)
model = SARIMAX(time_series, order = pdq, seasonal_order = PDQS)

#тренируем модель и делаем предсказание
model_fit = model.fit()
predicted = model_fit.predict(start=1, end=len(time_series) + 10) #метод predict принимет два параметра - когда начать предсказывать и когда пора остановиться. Здесь мы делаем предсказание на 10 месяцев вперед

#смотрим на наше предсказание
plt.plot(predicted, color = 'red')
plt.show()

