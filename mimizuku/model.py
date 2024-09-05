from mimizuku.rules.audit_command import AuditCommand
from mimizuku.rules.fs_notify import FsNotify


class Mimizuku:
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
        abuse_files=[],
        ignore_files=[],
        ignore_users=[],
    ):
        self.rules = [
            FsNotify(
                n_neighbors=n_neighbors,
                contamination=contamination,
                ignore_files=ignore_files,
                ignore_users=ignore_users,
                abuse_files=abuse_files,
            ),
            AuditCommand(
                n_neighbors=n_neighbors,
                contamination=contamination,
                ignore_users=ignore_users,
            ),
        ]

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
        ignore_files=[],
        abuse_files=[],
        ignore_users=[],
    ):
        fs_notify = FsNotify.load_model(
            model_path,
            ignore_files=ignore_files,
            abuse_files=abuse_files,
            ignore_users=ignore_users,
        )
        audit_command = AuditCommand.load_model(model_path, ignore_users=ignore_users)

        me = Mimizuku()
        me.rules = [fs_notify, audit_command]
        return me
