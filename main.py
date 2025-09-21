# %%


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %%
boston = fetch_openml(name="boston", version=1)
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name="Price")


# %%
features = ["RM", "LSTAT"]  # Example: RM (rooms), LSTAT (% lower status)
X_selected = X[features]


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)


# %%
model = LinearRegression()
model.fit(X_train, y_train)


# %%
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse}, MAE: {mae}")


# %%
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()



