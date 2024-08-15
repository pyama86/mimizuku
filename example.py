from mimizuku import Mimizuku

# Initialize the model
model = Mimizuku()

# Train the model with a log file or DataFrame
model.fit("./training.json")

# Save the trained model
model.save_model("./model.pkl")

# Load the model and use it for prediction
loaded_model = Mimizuku.load_model("./model.pkl")
anomalies_df = loaded_model.predict("./test.json")

# Display detected anomalies
print(anomalies_df)
