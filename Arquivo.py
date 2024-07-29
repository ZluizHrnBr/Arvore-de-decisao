import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import graphviz

# Carregando os dados do dataset do trabalho analisado
dev_data = pd.read_csv('dataset.csv')

# Seleção de características para a alocação de desenvolvedores em projetos
X = dev_data[['followers', 'NoC', 'DiP']]
y = dev_data['project']

# Divisão dos dados para treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo de árvore de decisão para o treinamento dos dados
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Treinamento do modelo de dados
fit = clf.fit(X_train, y_train)
# Previsão dos dados para teste
y_pred = clf.predict(X_test)
#Gerando uma árvore de decisão após o fit
dot_data = export_graphviz(fit, filled=True, out_file=None)
graph = graphviz.Source(dot_data)

# Avaliação do modelo feito com a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy:.3f}')
print(f'previsão para a alocação de recursos {y_pred}')

# Matriz de confusão após o treinamento e a predição dos dados
conf_m = confusion_matrix(y_test, y_pred)
print('Matriz de confusão')
print(conf_m)

#Amostragem dos dados obtidos após a predição e o treinamento dos dado utilizados pelos datasets
plt.imshow(conf_m, interpolation='nearest')
plt.title('Matriz de confusão')
plt.show()