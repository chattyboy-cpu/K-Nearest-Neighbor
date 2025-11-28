# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  
import numpy as np
import matplotlib.pyplot as plt
# loading dataset
df = pd.read_csv(r"C:\Users\lamin\OneDrive\Desktop\Datafolder\knn\train_set.csv")
y = df['Churn'].values
x= df[["Customer service calls","Account length"]].values

# creating a knn classifier
knn = KNeighborsClassifier(n_neighbors=6)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create neighbors
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)
  
    # Fit the model
    knn.fit(x_train, y_train)
  
    # Compute accuracy
    train_accuracies[neighbor] = knn.score(x_train, y_train)
    test_accuracies[neighbor] = knn.score(x_test, y_test)

print(neighbors, '\n', train_accuracies, '\n', test_accuracies)

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
# We convert .values() to a list so matplotlib can read it matches the shape of 'neighbors'
plt.plot(neighbors, list(train_accuracies.values()), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, list(test_accuracies.values()), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()