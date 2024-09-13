import pandas as pd

from mimizuku import Mimizuku
from mimizuku.rules.audit_command import AuditCommand
from mimizuku.rules.fs_notify import FsNotify

# Initialize the model
n_neighbors = 5
contamination = 0.001
ignore_user_names = ["root"]

fsn = FsNotify(
    n_neighbors=n_neighbors,
    contamination=contamination,
)
ac = AuditCommand(
    n_neighbors=n_neighbors,
    contamination=contamination,
    ignore_user_names=ignore_user_names,
)

model = Mimizuku()
model.add_rule(fsn)
model.add_rule(ac)

# Train the model with a log file or DataFrame
model.fit("./training.json")

# Save the trained model
model.save_model("./models")

# Load the model and use it for prediction
loaded_model = Mimizuku.load_model("./models")
anomalies_df = loaded_model.predict("./test.json")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Display detected anomalies
print("Detected anomalies:")
print(anomalies_df)
