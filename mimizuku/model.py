import re

import joblib
import orjson as json
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder


class Mimizuku:
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
        abuse_files=[],
        ignore_files=[],
        ignore_effective_users=[],
    ):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.event_encoder = LabelEncoder()
        self.ignore_files = ignore_files
        self.vectorizer = HashingVectorizer(
            n_features=2**20,
        )
        self.abuse_files = abuse_files
        self.ignore_effective_users = ignore_effective_users

    def replace_temp_strings(self, path):
        pattern = r"[a-f0-9]{7,40}"
        modified_path = re.sub(pattern, "", path)
        modified_path = re.sub(r"[\d]+", "", modified_path)
        return modified_path

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

            if user is not None and user in self.ignore_effective_users:
                return False
            return True

    def filename_to_vector(self, filename):
        return self.vectorizer.transform([filename])

    def load_and_preprocess(self, data, keep_original=False, fit=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
                        if not re.search(r'"id":\s*"55[0-9]"', line):
                            continue

                        alert = json.loads(line)
                        if self.is_target_event(alert):
                            alerts.append(alert)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        elif isinstance(data, dict):
            if self.is_target_event(data):
                alerts.append(data)
        else:
            raise ValueError("Input data must be a filepath or a DataFrame")

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
            if keep_original:
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

    def fit(self, data):
        try:
            X_train, _ = self.load_and_preprocess(data, fit=True)
            print("Fitting the model...")
            self.model.fit(X_train)
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def predict(self, data):
        try:
            X_test, df_test = self.load_and_preprocess(data, keep_original=True)
            print("Predicting anomalies...")
            anomalies = self.model.predict(X_test)

            anomalies_df = df_test[anomalies == -1]
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
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def save_model(self, model_path):
        joblib.dump(
            {
                "model": self.model,
                "event_encoder": self.event_encoder,
                "vectorizer": self.vectorizer,
            },
            model_path,
        )

    @staticmethod
    def load_model(
        model_path,
        ignore_files=[],
        abuse_files=[],
        ignore_effective_users=[],
    ):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku(
            ignore_files=ignore_files,
            abuse_files=abuse_files,
            ignore_effective_users=ignore_effective_users,
        )
        mimizuku.model = saved_objects["model"]
        mimizuku.event_encoder = saved_objects["event_encoder"]
        mimizuku.vectorizer = saved_objects["vectorizer"]
        return mimizuku
