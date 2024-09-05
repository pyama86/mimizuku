import re

import joblib
import pandas as pd

from .base import Base


class FsNotify(Base):
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
        ignore_files=[],
        ignore_users=[],
        abuse_files=[],
    ):
        self.ignore_users = ignore_users
        self.ignore_files = ignore_files
        self.abuse_files = abuse_files
        super().__init__(n_neighbors=n_neighbors, contamination=contamination)

    def name(self):
        return "fs_notify"

    def is_target_event(self, alert):
        if (
            "syscheck" in alert
            and re.match(r"55[0-9]", str(alert["rule"]["id"]))
            and int(alert["rule"]["level"]) > 0
            and all(
                not alert["syscheck"]["path"].startswith(ignore_file)
                for ignore_file in self.ignore_files
            )
        ):
            user = None
            if (
                alert["syscheck"]
                .get("audit", {})
                .get("effective_user", {})
                .get("name", None)
                is not None
            ):
                user = alert["syscheck"]["audit"]["effective_user"]["name"]
            elif alert["syscheck"].get("uname_after", None) is not None:
                user = alert["syscheck"]["uname_after"]

            if user is not None and user in self.ignore_users:
                return False
            return True

    def replace_temp_strings(self, path):
        pattern = r"[a-f0-9]{7,40}"
        modified_path = re.sub(pattern, "", path)
        modified_path = re.sub(r"[\d]+", "", modified_path)
        return modified_path

    def process_alerts(self, alerts, fit=False):
        original_data = []
        filenames = []
        for alert in alerts:
            syscheck = alert.get("syscheck", {})
            path = syscheck.get("path")
            event = syscheck.get("event", "")
            effective_user = (
                syscheck.get("audit", {}).get("effective_user", {}).get("name", "")
            )
            filenames.append(
                re.sub(
                    r"[\.\-_/]",
                    " ",
                    self.replace_temp_strings(path),
                )
            )

            if not fit:
                print(
                    f"hostname: {alert.get('agent', {}).get('name')} path: {path} event: {event} effective_user: {effective_user}"
                )

                original_data.append(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_event": event,
                        "original_effective_user": effective_user,
                    }
                )

        if len(filenames) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            X = self.vectorizer.fit_transform(filenames)
        else:
            X = self.vectorizer.transform(filenames)

        df = pd.DataFrame(original_data)
        return X, df

    def fill_anomaly_data(self, anomalies_df, df_test):
        for abuse_file in self.abuse_files:
            additional_anomalies = df_test[
                df_test["original_path"].str.contains(abuse_file)
            ]
            anomalies_df = pd.concat(
                [anomalies_df, additional_anomalies]
            ).drop_duplicates()

        return anomalies_df[
            [
                "original_hostname",
                "original_path",
                "original_event",
                "original_effective_user",
            ]
        ]

    @staticmethod
    def load_model(
        model_dir,
        ignore_files=[],
        abuse_files=[],
        ignore_users=[],
    ):
        me = FsNotify(
            ignore_files=ignore_files,
            abuse_files=abuse_files,
            ignore_users=ignore_users,
        )

        saved_objects = joblib.load(FsNotify.model_path(model_dir, me.name()))
        me.model = saved_objects["model"]
        me.vectorizer = saved_objects["vectorizer"]

        return me

    def filter_line(self, line):
        return not re.search(r'"id":\s*"55[0-9]"', line)
