import os
import re

import joblib
import pandas as pd

from .base import Base


class AuditCommand(Base):
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
        ignore_user_names=[],
        ignore_user_ids=[],
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.ignore_user_names = ignore_user_names
        self.ignore_user_ids = ignore_user_ids

    def save_model_extra(self):
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "ignore_user_names": self.ignore_user_names,
            "ignore_user_ids": self.ignore_user_ids,
        }

    def is_target_event(self, alert):
        ret = (
            "80792" == str(alert["rule"]["id"])
            and not any(
                f' UID="{user}"' in alert["full_log"] for user in self.ignore_user_names
            )
            and not any(
                f' uid="{user}"' in alert["full_log"] for user in self.ignore_user_ids
            )
        )
        return ret

    def process_alerts(self, alerts, fit=False):
        original_data = []
        commands = []
        for alert in alerts:
            command = " ".join(
                alert.get("data", {}).get("audit", {}).get("execve", {}).values()
            )
            commands.append(command)

            if not fit:
                user = None
                match = re.search(r' UID="(\w+)"', alert["full_log"])
                if match:
                    user = match.group(1)

                print(
                    f"hostname: {alert.get('agent', {}).get('name')} command: {command} user: {user}"
                )
                original_data.append(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_command": command,
                        "original_user": user,
                    }
                )

        if len(commands) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            X = self.vectorizer.fit_transform(commands)
        else:
            X = self.vectorizer.transform(commands)

        df = pd.DataFrame(original_data)
        return X, df

    def fill_anomaly_data(self, anomalies_df, df_test):
        return anomalies_df[
            [
                "original_hostname",
                "original_command",
                "original_user",
            ]
        ]

    @staticmethod
    def load_model(
        model_dir,
    ):
        if not os.path.exists(AuditCommand.model_path(model_dir, "audit_command")):
            return None

        saved_objects = joblib.load(AuditCommand.model_path(model_dir, "audit_command"))
        me = AuditCommand(
            n_neighbors=saved_objects["n_neighbors"],
            contamination=saved_objects["contamination"],
            ignore_user_names=saved_objects["ignore_user_names"],
            ignore_user_ids=saved_objects["ignore_user_ids"],
        )
        me.model = saved_objects["model"]
        me.vectorizer = saved_objects["vectorizer"]
        return me

    def filter_line(self, line):
        return not re.search(r'"id":\s*"80792"', line)
