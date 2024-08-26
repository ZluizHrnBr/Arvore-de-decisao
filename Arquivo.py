import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# Carregando os dados do dataset do trabalho analisado
dev_data = pd.read_csv('dataset.csv')

# Seleção de características para a alocação de desenvolvedores em projetos
label = LabelEncoder()

dev_data['job_number'] = label.fit_transform(dev_data['job'])

X = dev_data[['job_number', 'NoC', 'DiP', 'followers', 'ICT']]

y = dev_data[['project']]

transformer_numeric = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, [0]),
        ('dog', categorical_transformer, [1])])

clf = Pipeline(steps=[
    ('classifier', DecisionTreeClassifier(random_state=42))
])


X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

fit = clf.fit(X, y)
tree = clf.named_steps['classifier']


predict = clf.predict(X)
accuracy = clf.score(X_test, y_test)
print(f'Acuracia: {accuracy:.3f}')
print(tree)
predict = clf.predict(X)
print(predict)

plt.figure(figsize=(50, 40))
plot_tree(tree, filled=True, rounded=True)
plt.title("Árvore de Decisão")
plt.show()