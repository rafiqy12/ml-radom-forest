from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "forest-cover-type-dataset.zip"
TARGET = "Cover_Type"

st.set_page_config(
    page_title="Klasifikasi Forest Cover Type",
    page_icon="🌲",
    layout="wide",
)


@st.cache_data(show_spinner="Membaca dataset...")
def load_data(max_rows: int) -> pd.DataFrame:
    """Read covtype.csv directly from the downloaded Kaggle ZIP archive."""
    return pd.read_csv(DATASET_PATH, compression="zip", nrows=max_rows)


@st.cache_resource(show_spinner="Melatih model Random Forest...")
def train_model(data: pd.DataFrame, test_size: float, n_estimators: int):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # Decision Tree digunakan sebagai pendekatan seleksi fitur ala C5.0.
    selector_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=12,
        random_state=42,
    )
    # Gunakan array NumPy secara konsisten agar tidak muncul peringatan nama fitur.
    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()
    selector_model.fit(X_train_array, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    X_train_selected = selector.transform(X_train_array)
    X_test_selected = selector.transform(X_test_array)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions)
    importance = (
        pd.DataFrame(
            {
                "Fitur": selected_features,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return metrics, matrix, importance, selected_features


st.title("🌲 Klasifikasi Forest Cover Type")
st.write(
    "Implementasi Random Forest dengan seleksi fitur menggunakan Decision Tree "
    "sebagai pendekatan metode C5.0."
)

if not DATASET_PATH.exists():
    st.error(
        "Dataset tidak ditemukan. Pastikan file "
        "`forest-cover-type-dataset.zip` ikut diunggah ke repository."
    )
    st.stop()

with st.sidebar:
    st.header("Pengaturan model")
    max_rows = st.slider(
        "Jumlah data yang digunakan",
        min_value=10_000,
        max_value=150_000,
        value=50_000,
        step=10_000,
        help="Nilai lebih besar dapat meningkatkan hasil, tetapi pelatihan lebih lama.",
    )
    test_size = st.slider(
        "Proporsi data pengujian",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )
    n_estimators = st.slider(
        "Jumlah pohon Random Forest",
        min_value=50,
        max_value=300,
        value=100,
        step=50,
    )
    run_training = st.button("Latih dan evaluasi model", type="primary")

try:
    df = load_data(max_rows)
except Exception as error:
    st.exception(error)
    st.stop()

overview_col1, overview_col2, overview_col3 = st.columns(3)
overview_col1.metric("Jumlah baris", f"{len(df):,}")
overview_col2.metric("Jumlah fitur", len(df.columns) - 1)
overview_col3.metric("Jumlah kelas", df[TARGET].nunique())

with st.expander("Lihat contoh dataset"):
    st.dataframe(df.head(100), use_container_width=True)

st.subheader("Distribusi kelas")
class_counts = df[TARGET].value_counts().sort_index()
st.bar_chart(class_counts)

if not run_training:
    st.info(
        "Atur parameter di sidebar, kemudian klik **Latih dan evaluasi model**."
    )
    st.stop()

metrics, matrix, importance, selected_features = train_model(
    df, test_size, n_estimators
)

st.subheader("Hasil evaluasi")
metric_columns = st.columns(4)
metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
metric_columns[1].metric("Precision", f"{metrics['precision']:.4f}")
metric_columns[2].metric("Recall", f"{metrics['recall']:.4f}")
metric_columns[3].metric("F1-score", f"{metrics['f1']:.4f}")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Confusion matrix")
    fig_matrix, ax_matrix = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_matrix,
    )
    ax_matrix.set_xlabel("Prediksi")
    ax_matrix.set_ylabel("Aktual")
    st.pyplot(fig_matrix)
    plt.close(fig_matrix)

with chart_col2:
    st.markdown("#### Feature importance")
    top_importance = importance.head(15).sort_values("Importance")
    fig_importance, ax_importance = plt.subplots(figsize=(7, 5))
    ax_importance.barh(top_importance["Fitur"], top_importance["Importance"])
    ax_importance.set_xlabel("Importance")
    ax_importance.set_ylabel("Fitur")
    st.pyplot(fig_importance)
    plt.close(fig_importance)

st.success(
    f"Model menggunakan {len(selected_features)} dari "
    f"{len(df.columns) - 1} fitur."
)
with st.expander("Daftar fitur yang dipilih"):
    st.write(", ".join(selected_features))
    st.dataframe(importance, use_container_width=True)
