import re

import joblib
import pandas as pd

from .base import Base


class AuditCommand(Base):
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
        ignore_users=[],
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        self.ignore_users = ignore_users

    def name(self):
        return "audit_command"

    def is_target_event(self, alert):
        ret = "80792" == str(alert["rule"]["id"]) and not any(
            f'UID="{user}"' in alert["full_log"] for user in self.ignore_users
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
                print(
                    f"hostname: {alert.get('agent', {}).get('name')} command: {command}"
                )
                original_data.append(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_command": command,
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
            ]
        ]

    @staticmethod
    def load_model(
        model_dir,
        ignore_files=[],
        abuse_files=[],
        ignore_users=[],
    ):
        me = AuditCommand(ignore_users=ignore_users)
        saved_objects = joblib.load(AuditCommand.model_path(model_dir, me.name()))
        me.model = saved_objects["model"]
        me.vectorizer = saved_objects["vectorizer"]
        return me

    def filter_line(self, line):
        return not re.search(r'"id":\s*"80792"', line)
