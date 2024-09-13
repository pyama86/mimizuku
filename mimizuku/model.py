from mimizuku.rules.audit_command import AuditCommand
from mimizuku.rules.fs_notify import FsNotify


class Mimizuku:
    def __init__(
        self,
    ):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def fit(self, data):
        try:
            for rule in self.rules:
                rule.fit(data)
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def predict(self, data):
        predictions = []
        try:
            for rule in self.rules:
                r = rule.predict(data)
                predictions.append(
                    {
                        "rule": rule.name(),
                        "predictions": r,
                    }
                )
            return predictions
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def save_model(self, model_path):
        for rule in self.rules:
            rule.save_model(model_path)

    @staticmethod
    def load_model(
        model_path,
    ):
        rules = []
        for rule in [FsNotify, AuditCommand]:
            m = rule.load_model(model_path)
            if m:
                rules.append(m)

        me = Mimizuku()
        me.rules = rules
        return me
