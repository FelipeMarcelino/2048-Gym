import optuna
import sys

study_name = sys.argv[1]
study = optuna.create_study(
    study_name=study_name,
    storage="postgresql://felipemarcelino:123456@localhost/database",
    load_if_exists=True,
    direction="maximize",
)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df["value"].max())
print(df)

