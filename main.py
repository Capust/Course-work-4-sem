import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score


# Загрузить данные
data = pd.read_csv('heart.csv')

# Разделить данные на функции (X) и целевую переменную (y)
X = data.drop('age', axis=1)
y = data['age']

# Разделить данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализировать модели
models = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Создание словаря для хранения результатов
results = {}

# Обучение и оценка моделей
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R2': r2}
    print(f'{model_name} MSE: {mse}')
    print(f'{model_name} R2: {r2}')
    print('-' * 60)

# Отобразить подробный отчет о классификации для лучшей модели
# Выбор лучшей модели
best_model_name = min(results, key=lambda x: results[x]['MSE'])
print(f'Best model: {best_model_name} with MSE of {results[best_model_name]["MSE"]} and R2 of {results[best_model_name]["R2"]}')

