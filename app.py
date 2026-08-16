from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.sav"
LOGO_PATH = APP_DIR / "logo.png"

FEATURE_COLUMNS = [
    "age",
    "job",
    "balance",
    "housing",
    "loan",
    "contact",
    "month",
    "campaign",
    "pdays",
    "poutcome",
]

JOB_OPTIONS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
YES_NO_OPTIONS = ["no", "yes"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]


st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1F3A 0%, #123A63 100%);
        border-right: 1px solid #0B1F3A;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.35rem;
        border-radius: 0.65rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.22);
    }
    .main-header {
        padding: 0.15rem 0 0.85rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.25rem;
    }
    .main-header h1 {
        margin: 0;
        color: #0B1F3A;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: Path):
    """Load the fitted preprocessing + XGBoost pipeline."""
    with model_path.open("rb") as file:
        return pickle.load(file)


def positive_class_index(model) -> int:
    """Return the probability-column index representing deposit=True/yes/1."""
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == 0 and hasattr(model, "named_steps"):
        classes = np.asarray(getattr(model.named_steps["model"], "classes_", []))

    for candidate in (True, 1, "yes", "true", "deposit"):
        matches = np.where(classes == candidate)[0]
        if matches.size:
            return int(matches[0])

    if classes.size == 2:
        return 1
    raise ValueError("Kelas positif model tidak dapat ditentukan.")


def normalize_prediction(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    return str(value).strip().lower() in {"yes", "true", "1", "deposit"}


def predict(model, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prediction = model.predict(data[FEATURE_COLUMNS])
    probabilities = model.predict_proba(data[FEATURE_COLUMNS])
    positive_probability = probabilities[:, positive_class_index(model)]
    labels = np.asarray([normalize_prediction(value) for value in prediction])
    return labels, positive_probability


def validate_batch(data: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        errors.append("Kolom wajib belum tersedia: " + ", ".join(missing))
        return errors

    allowed_values = {
        "job": set(JOB_OPTIONS),
        "housing": set(YES_NO_OPTIONS),
        "loan": set(YES_NO_OPTIONS),
        "contact": set(CONTACT_OPTIONS),
        "month": set(MONTH_OPTIONS),
        "poutcome": set(POUTCOME_OPTIONS),
    }
    for column, allowed in allowed_values.items():
        invalid = sorted(set(data[column].dropna().astype(str)) - allowed)
        if invalid:
            errors.append(
                f"Nilai tidak dikenali pada {column}: {', '.join(invalid[:8])}"
            )
    return errors


def sample_template() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "age": 35,
            "job": "management",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "campaign": 2,
            "pdays": -1,
            "poutcome": "unknown",
        }]
    )


def prediction_records(
    data: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    source: str,
) -> pd.DataFrame:
    """Build history rows without requiring customer identity fields."""
    records = data[FEATURE_COLUMNS].copy().reset_index(drop=True)
    records.insert(0, "prediction_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    records.insert(1, "source", source)
    records["prediction"] = np.where(labels, "Potential", "Not Potential")
    records["probability_deposit"] = probabilities.astype(float)
    records["priority"] = pd.cut(
        records["probability_deposit"],
        bins=[-np.inf, 0.30, 0.50, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return records


def add_to_history(records: pd.DataFrame) -> None:
    """Append prediction records and assign a stable sequential number."""
    history = st.session_state.prediction_history.copy()
    next_number = int(history["record_no"].max()) + 1 if not history.empty else 1
    records = records.copy()
    records.insert(0, "record_no", range(next_number, next_number + len(records)))
    st.session_state.prediction_history = pd.concat(
        [history, records], ignore_index=True
    )


HISTORY_COLUMNS = [
    "record_no", "prediction_time", "source", *FEATURE_COLUMNS,
    "prediction", "probability_deposit", "priority",
]

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=HISTORY_COLUMNS)


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)
    else:
        st.markdown("## 🏦 Bank Deposit")
        st.caption("Tambahkan `logo.png` di folder aplikasi.")

    st.markdown("### Prediction Dashboard")
    st.markdown("---")
    selected_menu = st.radio(
        "Menu Utama",
        options=["Prediksi Customer", "Riwayat & Analisis"],
        format_func=lambda value: {
            "Prediksi Customer": "🎯  Prediksi Customer",
            "Riwayat & Analisis": "📊  Riwayat & Analisis",
        }[value],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Creator: Aris Sando Hamzah")

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Bank Deposit Prediction</h1>
        <p>Aplikasi machine learning untuk menentukan prioritas target campaign deposito.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "File `best_model.sav` belum ditemukan. Tempatkan file model di folder "
        "yang sama dengan `app.py`, lalu jalankan kembali aplikasi."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.info(
        "Pastikan versi Python, scikit-learn, XGBoost, category-encoders, "
        "dan CatBoost kompatibel dengan environment saat model disimpan."
    )
    st.stop()

if selected_menu == "Prediksi Customer":
    st.subheader("Prediksi Customer")
    st.caption(
        "Masukkan satu profil customer atau unggah beberapa data sekaligus. "
        "Hasil prediksi dapat dibandingkan pada menu Riwayat & Analisis."
    )
    single_tab, batch_tab = st.tabs(["👤 Satu Customer", "📄 Unggah CSV"])

    with single_tab:
        with st.form("single_prediction_form"):
            st.subheader("Profil Customer")
            left, middle, right = st.columns(3)
    
            with left:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                job = st.selectbox("Job", JOB_OPTIONS, index=4)
                balance = st.number_input("Balance", value=1500, step=100)
                housing = st.selectbox("Housing Loan", YES_NO_OPTIONS, index=1)
    
            with middle:
                loan = st.selectbox("Personal Loan", YES_NO_OPTIONS)
                contact = st.selectbox("Contact", CONTACT_OPTIONS)
                month = st.selectbox("Month", MONTH_OPTIONS, index=4)
    
            with right:
                campaign = st.number_input(
                    "Campaign (jumlah kontak)", min_value=1, value=2, step=1
                )
                pdays = st.number_input(
                    "Pdays (-1 = belum pernah dihubungi)", min_value=-1,
                    value=-1, step=1
                )
                poutcome = st.selectbox("Previous Outcome", POUTCOME_OPTIONS, index=3)
    
            submitted = st.form_submit_button(
                "Prediksi Customer", type="primary", use_container_width=True
            )
    
        if submitted:
            input_data = pd.DataFrame(
                [{
                    "age": age,
                    "job": job,
                    "balance": balance,
                    "housing": housing,
                    "loan": loan,
                    "contact": contact,
                    "month": month,
                    "campaign": campaign,
                    "pdays": pdays,
                    "poutcome": poutcome,
                }]
            )
    
            try:
                labels, probabilities = predict(model, input_data)
                probability = float(probabilities[0])
                add_to_history(
                    prediction_records(input_data, labels, probabilities, "Single")
                )
                metric_col, result_col = st.columns([1, 2])
                metric_col.metric("Probabilitas Deposit", f"{probability:.2%}")
    
                if labels[0]:
                    result_col.success(
                        "Customer diprediksi potensial membuka deposito. "
                        "Prioritaskan untuk campaign."
                    )
                else:
                    result_col.warning(
                        "Customer diprediksi belum potensial membuka deposito."
                    )
    
                with st.expander("Lihat data input"):
                    st.dataframe(input_data, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Prediksi gagal: {exc}")

    with batch_tab:
        st.subheader("Prediksi Banyak Customer")
        st.write(
            "Unggah CSV dengan 10 kolom input mentah. Kolom tambahan akan "
            "dipertahankan pada hasil unduhan."
        )
        st.download_button(
            "Unduh Template CSV",
            data=sample_template().to_csv(index=False).encode("utf-8"),
            file_name="template_customer.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"CSV gagal dibaca: {exc}")
            else:
                errors = validate_batch(batch_data)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        labels, probabilities = predict(model, batch_data)
                        result = batch_data.copy()
                        result["prediction_deposit"] = np.where(labels, "yes", "no")
                        result["probability_deposit"] = probabilities
    
                        total = len(result)
                        potential = int(labels.sum())
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Customer", f"{total:,}")
                        c2.metric("Customer Potensial", f"{potential:,}")
                        c3.metric(
                            "Proporsi Potensial",
                            f"{potential / total:.2%}" if total else "0.00%",
                        )
    
                        st.dataframe(
                            result.sort_values(
                                "probability_deposit", ascending=False
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Unduh Hasil Prediksi",
                            data=result.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_prediksi_deposito.csv",
                            mime="text/csv",
                            type="primary",
                        )
    
                        if st.button(
                            "Tambahkan ke Riwayat",
                            type="secondary",
                            use_container_width=True,
                        ):
                            add_to_history(
                                prediction_records(
                                    batch_data, labels, probabilities, "CSV"
                                )
                            )
                            st.success(
                                f"{len(batch_data):,} hasil prediksi ditambahkan "
                                "ke riwayat."
                            )
                    except Exception as exc:
                        st.error(f"Prediksi batch gagal: {exc}")

else:
    st.subheader("Riwayat Prediksi")
    st.caption(
        "Riwayat dibedakan berdasarkan nomor otomatis dan waktu prediksi. "
        "Data tersimpan selama sesi aplikasi masih aktif."
    )

    history = st.session_state.prediction_history.copy()

    if history.empty:
        st.info(
            "Belum ada riwayat. Lakukan prediksi satu customer atau tambahkan "
            "hasil prediksi CSV."
        )
    else:
        history["probability_deposit"] = pd.to_numeric(
            history["probability_deposit"], errors="coerce"
        )

        total_history = len(history)
        total_potential = int((history["prediction"] == "Potential").sum())
        average_probability = history["probability_deposit"].mean()
        high_priority = int((history["priority"] == "High").sum())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Prediksi", f"{total_history:,}")
        kpi2.metric("Customer Potensial", f"{total_potential:,}")
        kpi3.metric("Rata-rata Probabilitas", f"{average_probability:.2%}")
        kpi4.metric("Prioritas Tinggi", f"{high_priority:,}")

        filter1, filter2 = st.columns(2)
        with filter1:
            prediction_filter = st.multiselect(
                "Filter hasil prediksi",
                options=["Potential", "Not Potential"],
                default=["Potential", "Not Potential"],
            )
        with filter2:
            priority_filter = st.multiselect(
                "Filter prioritas",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

        filtered = history[
            history["prediction"].isin(prediction_filter)
            & history["priority"].isin(priority_filter)
        ].copy()

        chart_left, chart_right = st.columns([2, 1])
        with chart_left:
            st.markdown("#### Perbandingan Probabilitas")
            probability_chart = (
                alt.Chart(filtered)
                .mark_circle(size=110, opacity=0.85)
                .encode(
                    x=alt.X("record_no:O", title="Nomor Prediksi"),
                    y=alt.Y(
                        "probability_deposit:Q",
                        title="Probabilitas Deposit",
                        axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    color=alt.Color(
                        "prediction:N",
                        title="Hasil",
                        scale=alt.Scale(
                            domain=["Potential", "Not Potential"],
                            range=["#22C55E", "#F59E0B"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("record_no:O", title="No."),
                        alt.Tooltip("prediction_time:N", title="Waktu"),
                        alt.Tooltip("probability_deposit:Q", format=".2%"),
                        alt.Tooltip("prediction:N", title="Hasil"),
                        alt.Tooltip("priority:N", title="Prioritas"),
                    ],
                )
                .properties(height=330)
                .interactive()
            )
            threshold = (
                alt.Chart(pd.DataFrame({"threshold": [0.5]}))
                .mark_rule(color="#EF4444", strokeDash=[6, 4])
                .encode(y="threshold:Q")
            )
            st.altair_chart(
                probability_chart + threshold, use_container_width=True
            )

        with chart_right:
            st.markdown("#### Komposisi Prediksi")
            composition = (
                filtered.groupby("prediction", as_index=False)
                .size()
                .rename(columns={"size": "total"})
            )
            donut = (
                alt.Chart(composition)
                .mark_arc(innerRadius=65, outerRadius=115)
                .encode(
                    theta=alt.Theta("total:Q"),
                    color=alt.Color(
                        "prediction:N",
                        title="Hasil",
                        scale=alt.Scale(
                            domain=["Potential", "Not Potential"],
                            range=["#22C55E", "#F59E0B"],
                        ),
                    ),
                    tooltip=["prediction:N", "total:Q"],
                )
                .properties(height=330)
            )
            st.altair_chart(donut, use_container_width=True)

        st.markdown("#### Tabel Prioritas Campaign")
        table = filtered.sort_values(
            ["probability_deposit", "record_no"], ascending=[False, True]
        ).copy()
        table.insert(0, "delete", False)

        edited_table = st.data_editor(
            table,
            use_container_width=True,
            hide_index=True,
            disabled=[column for column in table.columns if column != "delete"],
            column_config={
                "delete": st.column_config.CheckboxColumn(
                    "Hapus", help="Centang data yang akan dihapus."
                ),
                "record_no": st.column_config.NumberColumn("No.", format="%d"),
                "prediction_time": "Waktu Prediksi",
                "source": "Sumber",
                "prediction": "Hasil",
                "priority": "Prioritas",
                "probability_deposit": st.column_config.ProgressColumn(
                    "Probabilitas Deposit",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
            },
            key="history_editor",
        )

        action1, action2, action3 = st.columns([1, 1, 2])
        with action1:
            if st.button("Hapus Data Terpilih", use_container_width=True):
                selected_numbers = edited_table.loc[
                    edited_table["delete"], "record_no"
                ].tolist()
                if selected_numbers:
                    st.session_state.prediction_history = history[
                        ~history["record_no"].isin(selected_numbers)
                    ].reset_index(drop=True)
                    st.success(f"{len(selected_numbers)} data berhasil dihapus.")
                    st.rerun()
                else:
                    st.warning("Centang minimal satu data yang ingin dihapus.")

        with action2:
            if st.button("Hapus Semua Riwayat", use_container_width=True):
                st.session_state.prediction_history = pd.DataFrame(
                    columns=HISTORY_COLUMNS
                )
                st.rerun()

        with action3:
            st.download_button(
                "Unduh Riwayat CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="riwayat_prediksi_deposito.csv",
                mime="text/csv",
                use_container_width=True,
            )
