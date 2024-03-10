import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the Titanic dataset
df = pd.read_csv("titanic_dataset.csv")

# Preprocess the dataset
# (Perform necessary data cleaning, feature engineering, and encoding)

# Split the dataset into features and target variable
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define and train the model
model = Sequential()
model.add(Dense(16, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("titanic_model.h5")
