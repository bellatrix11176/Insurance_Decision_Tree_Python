from pathlib import Path
import argparse

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text


# ============================================================
# UTILITIES
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def prettify_tree(tree_text: str) -> str:
    """
    Converts one-hot splits like:
      Payment Method_Credit Card <= 0.5
    into:
      Payment Method != Credit Card
    """
    out = []
    for line in tree_text.splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        content = line.strip()

        if ("<=" in content or ">" in content) and "_" in content:
            parts = content.split()
            replaced = False

            for i, p in enumerate(parts):
                if p in ("<=", ">"):
                    feature = " ".join(parts[:i])
                    op = parts[i]

                    try:
                        thresh = float(parts[i + 1])
                    except Exception:
                        break

                    # For one-hot encoded features, threshold is typically 0.5
                    if abs(thresh - 0.5) < 1e-6 and "_" in feature:
                        base, category = feature.split("_", 1)
                        if op == "<=":
                            out.append(f"{indent}{base} != {category}")
                        else:
                            out.append(f"{indent}{base} = {category}")
                        replaced = True
                    break

            if not replaced:
                out.append(line)
        else:
            out.append(line)

    return "\n".join(out)


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """
    Builds a preprocessing + decision tree pipeline.
    """
    # NOTE: `sparse_output=False` is for newer scikit-learn versions.
    # If you ever see an error, switch to: OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    tree = DecisionTreeClassifier(random_state=42)

    return Pipeline(steps=[("prep", preprocessor), ("tree", tree)])


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a decision tree on insurance data and score new applicants."
    )

    # Repo-root defaults (portable). Users can override with flags if needed.
    repo_root = Path(__file__).resolve().parents[1]
    default_excel = repo_root / "data" / "InsuranceData.xlsx"
    default_out_dir = repo_root / "outputs"

    parser.add_argument("--excel", type=Path, default=default_excel, help="Path to InsuranceData.xlsx")
    parser.add_argument("--outdir", type=Path, default=default_out_dir, help="Output directory")

    parser.add_argument("--train-sheet", default="Issued Policies", help="Excel sheet name for training data")
    parser.add_argument("--score-sheet", default="New Applicants", help="Excel sheet name for scoring data")
    parser.add_argument("--label-col", default="Insurance Category", help="Target/label column name")

    args = parser.parse_args()

    excel_file: Path = args.excel
    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_txt = out_dir / "InsuranceDecisionTree_MenuTree.txt"
    scored_csv = out_dir / "InsuranceData_NewApplicants_scored_by_python.csv"

    print("\n================ LOADING DATA =================")
    print("Excel file:", excel_file)

    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file at:\n  {excel_file}\n\n"
            "Expected location (default): data/InsuranceData.xlsx\n"
            "You can also pass a custom path using: --excel path/to/InsuranceData.xlsx"
        )

    train = pd.read_excel(excel_file, sheet_name=args.train_sheet).dropna(how="all")
    score = pd.read_excel(excel_file, sheet_name=args.score_sheet).dropna(how="all")

    train = clean_columns(train)
    score = clean_columns(score)

    print("Training shape:", train.shape)
    print("Scoring shape :", score.shape)

    if args.label_col not in train.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    y = train[args.label_col].astype(str)
    X = train.drop(columns=[args.label_col])

    missing = set(X.columns) - set(score.columns)
    if missing:
        raise ValueError(f"Scoring data missing columns: {sorted(missing)}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print("\nNumeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    print("\n================ TRAINING TREE =================")
    pipeline.fit(X, y)

    encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
    encoded_cat_features = encoder.get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + encoded_cat_features.tolist()

    raw_tree = export_text(
        pipeline.named_steps["tree"],
        feature_names=feature_names,
        decimals=3
    )

    pretty_tree = prettify_tree(raw_tree)
    tree_txt.write_text(pretty_tree, encoding="utf-8")

    print("\nTree written to:")
    print(tree_txt)

    print("\n================ SCORING NEW APPLICANTS =================")
    probabilities = pipeline.predict_proba(score[X.columns])
    predictions = pipeline.predict(score[X.columns])
    classes = pipeline.classes_

    scored = score.copy()
    scored[f"prediction({args.label_col})"] = predictions

    for i, cls in enumerate(classes):
        scored[f"confidence({cls})"] = probabilities[:, i]

    scored.to_csv(scored_csv, index=False)

    print("Scored data saved to:")
    print(scored_csv)

    print("\n================ PREVIEW =================")
    cols = [f"prediction({args.label_col})"] + [f"confidence({c})" for c in classes]
    print(scored[cols].head(8))

    print("\n✅ DONE. Open outputs/InsuranceDecisionTree_MenuTree.txt to view the full decision tree.")


if __name__ == "__main__":
    main()
