# Mimizuku: Anomaly Detection for Wazuh Alerts

Mimizuku is a Python library designed for anomaly detection based on filesystem and command auditing events extracted from Wazuh alerts. It leverages unsupervised learning techniques to identify unusual activities in system logs, making it ideal for security-related use cases, such as detecting unauthorized file modifications or suspicious command executions.

## Features

- **Wazuh Alerts Integration**: Specifically designed to process Wazuh alert logs for anomaly detection.
- **Flexible Anomaly Detection**: Detects anomalies using filesystem events and command executions based on custom rules.
- **Customizable Settings**: Configure parameters such as the number of neighbors, contamination rate, and file/user ignore lists.
- **Filesystem Event Monitoring**: Automatically identifies suspicious file changes.
- **Command Auditing**: Detects anomalies in command execution patterns.
- **Model Persistence**: Easily save and load trained models for future use.

## Installation

```bash
pip install mimizuku
```

## Usage

### 1. Initialize and Train the Model

```python
import pandas as pd
from mimizuku import Mimizuku

# Initialize the model with custom settings
model = Mimizuku(contamination=0.001, n_neighbors=5)

# Train the model using a Wazuh alert log file or DataFrame
model.fit("./training.json")

# Save the trained model for later use
model.save_model("./models")
```

### 2. Load and Use the Model for Anomaly Detection

```python
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

# Display detected anomalies
print("Detected anomalies:")
print(anomalies_df)
```

## Customization Options

### Parameters for Model Initialization:
- **`n_neighbors`**: Number of neighbors to use for the Local Outlier Factor algorithm.
- **`contamination`**: Proportion of the dataset that is expected to be anomalous.

### Model Persistence:
- **`save_model(model_path)`**: Saves the trained model and vectorizer to a specified path.
- **`load_model(model_path)`**: Loads a saved model and applies ignore lists during prediction.

## Example Log Format

The input data for the model is expected to be in JSON format, following the Wazuh alert structure. Below is an example of a Wazuh alert log entry that Mimizuku can process:

```json
{
  "syscheck": {
    "path": "/etc/passwd",
    "event": "modified",
    "audit": {
      "effective_user": {
        "name": "root"
      }
    }
  },
  "agent": {
    "name": "my-hostname"
  },
  "rule": {
    "id": "550",
    "level": 7
  }
}
```

## License

Mimizuku is licensed under the MIT License.
