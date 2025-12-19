import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

x = df["sepal length (cm)"]
y = df["petal length (cm)"]

sns.regplot(x=x, y=y)
plt.title("Correlation: Sepal Length vs Petal Length")
plt.savefig("correlation.png")
plt.show()

