import pandas as pd

from mimizuku import Mimizuku

# Initialize the model
model = Mimizuku(contamination=0.01, n_neighbors=20)

# Train the model with a log file or DataFrame
model.fit("./training.json")

# Save the trained model
model.save_model("./model.pkl")

# Load the model and use it for prediction
loaded_model = Mimizuku.load_model("./model.pkl")
anomalies_df = loaded_model.predict("./test.json")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Display detected anomalies
print("Detected anomalies:")
print(anomalies_df)
