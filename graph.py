import pandas as pd
import matplotlib.pyplot as plt

# Read csv into dataframe
df = pd.read_csv("results/model_results.csv")

# Accuracy graph
plt.plot(df["train_accuracy"], label = "train accuracy")
plt.plot(df["test_accuracy"], label = "test accuracy")
plt.title("Model Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./graph/model_Accuracy.jpg")
plt.show()

# Loss graph
plt.plot(df["train_loss"], label = "train loss")
plt.plot(df["test_loss"], label = "test loss")
plt.title("Model Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./graph/model_Loss.jpg")
plt.show()