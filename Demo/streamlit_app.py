from pathlib import Path
import socket
import subprocess
import sys
import time
import webbrowser

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
NBFO_DIR = PROJECT_ROOT / "NBFO_IB"
CLUSTER_DIR = PROJECT_ROOT / "Cluster_nonIB"

NBFO_INPUT_PATH = NBFO_DIR / "processed_data" / "gcon_model_input.parquet"
NBFO_MODEL_PATHS = {
    "XGBoost": NBFO_DIR / "saved_models" / "selected_xgboost_model.joblib",
    "LightGBM": NBFO_DIR / "saved_models" / "lightgbm_best_model.joblib",
}
NBFO_SCORE_PATHS = {
    "propensity_in_h": NBFO_DIR / "saved_models" / "gcon_test_scores_best_xgboost.parquet",
    "propensity_calibrated": NBFO_DIR / "saved_models" / "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet",
}

CLUSTER_PERSONA_PATH = CLUSTER_DIR / "output" / "nonib_final_personas.parquet"
CLUSTER_SUMMARY_PATH = CLUSTER_DIR / "output" / "nonib_cluster_summary.csv"
CLUSTER_RECOMMENDATION_PATH = CLUSTER_DIR / "output" / "nonib_recommendations.csv"

ID_COLS = ["CUSTOMER_NUMBER", "PRODUCT_CODE", "PRODUCT_NAME", "MONTH"]
TARGET_COL = "SUBSCRIPTION"
OWN_COLS = [
    "OWN_CURRENT_ACCOUNT",
    "OWN_TERM_DEPOSIT",
    "OWN_CREDIT_CARD",
    "OWN_DEBIT_CARD",
    "OWN_LENDING",
]
PRODUCT_TO_OWN = {
    "CURRENT_ACCOUNT": "OWN_CURRENT_ACCOUNT",
    "TERM_DEPOSIT": "OWN_TERM_DEPOSIT",
    "CREDIT_CARD": "OWN_CREDIT_CARD",
    "DEBIT_CARD": "OWN_DEBIT_CARD",
    "LENDING": "OWN_LENDING",
}
CLUSTER_PERSONA_DISPLAY = {
    -1: "Dormant / Ngủ đông",
    0: "Cluster 0: Traditional",
    1: "Cluster 1: High-Value Traditional",
    2: "Cluster 2: Senior High-Value Heavy Borrower",
    3: "Cluster 3: Senior Ultra TD Saver",
    4: "Cluster 4: Senior Multi-product Saver",
    5: "Cluster 5: High-Value Saver",
    6: "Cluster 6: Stable Senior Saver",
    7: "Cluster 7: High-Value Heavy Borrower",
}
PERSONA_APPROACH = {
    -1: (
        "Ít hoạt động, ít sản phẩm. Không ưu tiên trong giai đoạn đầu, dồn nguồn lực cho nhóm có tín hiệu rõ hơn."
    ),
    0: (
        "Nhóm phổ thông lớn nhất, chủ yếu chỉ có tài khoản thanh toán và một phần thẻ ghi nợ. "
        "Onboarding nhẹ: gợi ý giao dịch cơ bản đầu tiên qua app như chuyển khoản, nạp điện thoại kèm hướng dẫn. "
        "Ưu tiên trung bình, số đông nhưng giá trị/đầu KH thấp."
    ),
    1: (
        "Nhóm giá trị cao nhưng hành vi còn truyền thống. Đã quen sản phẩm tín dụng, gợi ý quản lý thẻ/khoản vay "
        "qua app là bước số hóa tự nhiên nhất."
    ),
    2: (
        "Lớn tuổi, dư nợ và tài sản cao. Tương tự nhóm High-Value Heavy Borrower nhưng kết hợp kênh chi nhánh "
        "do yếu tố tuổi."
    ),
    3: (
        "Khách hàng lớn tuổi, tập trung rất mạnh vào tiền gửi kỳ hạn. TD balance cực cao, gần như không vay, "
        "số lượng sản phẩm thấp. Phù hợp với thông điệp an toàn, đơn giản và quản lý sổ tiết kiệm."
    ),
    4: (
        "Khách hàng lớn tuổi, giá trị tiết kiệm cao nhưng có quan hệ đa sản phẩm hơn. Có yếu tố vay, nên ưu tiên "
        "quản lý tổng quan tài sản, khoản vay và tiền gửi qua app hoặc qua RM."
    ),
    5: (
        "Nhóm saver giá trị cao, quy mô lớn. Tiền chủ yếu nằm ở tiền gửi kỳ hạn, gần như không vay. Tiếp cận bằng "
        "dịch vụ thông tin sổ, theo dõi lãi suất và quản lý tiền gửi qua app."
    ),
    6: (
        "Khách hàng lớn tuổi, tiền gửi cao và ổn định. Không vay, quy mô lớn hơn các nhóm senior saver nhỏ, sử dụng "
        "khoảng 2 sản phẩm. Phù hợp với hướng duy trì tiền gửi và mở rộng sử dụng kênh số từng bước."
    ),
    7: (
        "Nhóm nhỏ nhưng giá trị cao, dư nợ rất lớn, dùng nhiều sản phẩm nhất. Cần chăm sóc cá nhân hóa, gợi ý quản lý "
        "khoản vay và thanh toán qua app."
    ),
}

PERSONA_CAMPAIGN = {
    -1: {
        "Chân dung": "Không có sản phẩm nào, gần như không hoạt động. Không có bất kỳ tín hiệu digital-ready nào.",
        "Hướng tiếp cận": "Không ưu tiên trong giai đoạn đầu. Dồn nguồn lực cho các nhóm có tín hiệu rõ hơn trước.",
        "Kênh & hành động đầu tiên": "Gửi liên lạc tối thiểu. Xem xét lại sau khi các nhóm ưu tiên cao đã được khai thác.",
    },
    0: {
        "Chân dung": "Nhóm lớn nhất, giá trị/đầu KH thấp, chủ yếu TK thanh toán + một phần thẻ ghi nợ. Quen giao dịch tại quầy.",
        "Hướng tiếp cận": "Automation quy mô lớn, chi phí thấp. Gợi ý giao dịch cơ bản đầu tiên qua app kèm incentive nhỏ.",
        "Kênh & hành động đầu tiên": "Push/SMS hàng loạt + deep-link, chuyển khoản hoặc nạp điện thoại lần đầu.",
    },
    1: {
        "Chân dung": "Gần nổi bật ở thẻ tín dụng & khoản vay - đã quen sản phẩm phức tạp nhưng vẫn dùng chi nhánh.",
        "Hướng tiếp cận": "Dùng thói quen quản lý thẻ/khoản vay làm cú hích - số hóa tự nhiên nhất với nhóm này.",
        "Kênh & hành động đầu tiên": "Chi nhánh + SMS gần kỳ thanh toán thẻ - thanh toán thẻ qua app lần đầu.",
    },
    2: {
        "Chân dung": "Lớn tuổi, tài sản + dư nợ cao. Cần chăm sóc cá nhân hóa & theo dõi rủi ro tín dụng.",
        "Hướng tiếp cận": "RM + chi nhánh kết hợp; không ép sang digital ngay - xây lòng tin trước, demo app tại chỗ.",
        "Kênh & hành động đầu tiên": "RM gặp mặt, demo xem lịch trả nợ qua app, để KH tự thử với hỗ trợ.",
    },
    3: {
        "Chân dung": "Lớn tuổi, TD balance cao, gần như không vay, số sản phẩm thấp - tập trung vào tích lũy.",
        "Hướng tiếp cận": "Thông điệp bảo toàn tài sản: theo dõi & nhận cảnh báo đáo hạn qua app thay vì phải đến quầy.",
        "Kênh & hành động đầu tiên": "SMS cảnh báo đáo hạn TD, nhân viên chi nhánh hỗ trợ cài app & xem sổ tiết kiệm.",
    },
    4: {
        "Chân dung": "Lớn tuổi, tiết kiệm cao, có thêm khoản vay và nhiều sản phẩm hơn C3.",
        "Hướng tiếp cận": "Tận dụng quan hệ đa sản phẩm để gợi ý quản lý tổng thể tài chính qua app - không chỉ xem tiết kiệm.",
        "Kênh & hành động đầu tiên": "Chi nhánh + SMS đề xem tổng quan tài sản & khoản vay qua app cùng lúc.",
    },
    5: {
        "Chân dung": "Tiết kiệm kỳ hạn cao, gần như không vay, hành vi tài chính đơn giản - chỉ chưa dùng kênh số.",
        "Hướng tiếp cận": "Tiếp cận bằng giá trị quản lý tài sản: xem sổ tiết kiệm, theo dõi lãi suất & cảnh báo đáo hạn qua app.",
        "Kênh & hành động đầu tiên": "SMS trước kỳ đáo hạn TD - cài App - Xem/gia hạn sổ tiết kiệm.",
    },
    6: {
        "Chân dung": "Lớn tuổi, tiền gửi cao & ổn định, gần như không vay. Quy mô lớn, dùng khoảng 2 sản phẩm.",
        "Hướng tiếp cận": "Chi nhánh làm cầu nối. Thông điệp an toàn, đơn giản - xem sổ tiết kiệm qua app không cần ra quầy.",
        "Kênh & hành động đầu tiên": "Nhân viên chi nhánh demo app khi KH đến đáo hạn - xem sổ dư tiết kiệm lần đầu.",
    },
    7: {
        "Chân dung": "Nhóm nhỏ nhưng giá trị cao nhất - tài sản lớn, dư nợ rất cao, đa sản phẩm nhất trong tất cả cluster.",
        "Hướng tiếp cận": "Chăm sóc RM cá nhân hóa, gợi ý quản lý khoản vay + thanh toán định kỳ qua app.",
        "Kênh & hành động đầu tiên": "RM gọi trực tiếp - Hỗ trợ cài app tại chỗ - Xem lịch & thanh toán khoản vay lần đầu.",
    },
}


st.set_page_config(page_title="GCON Demo", layout="wide")


def format_number(value):
    if pd.isna(value):
        return "-"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def first_existing(row, columns, default=np.nan):
    for col in columns:
        if col in row.index:
            return row[col]
    return default


def display_persona_name(row):
    return str(first_existing(row, ["PERSONA_NAME"], "Unknown"))


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return joblib.load(NBFO_MODEL_PATHS[model_name])


@st.cache_data(show_spinner="Loading NBFO scored candidates...")
def load_nbfo_data():
    score_frames = []
    for score_col, score_path in NBFO_SCORE_PATHS.items():
        frame = pd.read_parquet(score_path)
        frame["MONTH"] = pd.to_datetime(frame["MONTH"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        frame = frame.rename(columns={"SUBSCRIPTION_PROPENSITY": score_col})
        score_frames.append(frame[ID_COLS + [TARGET_COL, score_col]])

    scores = score_frames[0]
    for frame in score_frames[1:]:
        merge_cols = [c for c in frame.columns if c not in ID_COLS and c != TARGET_COL]
        scores = scores.merge(frame[ID_COLS + merge_cols], on=ID_COLS, how="left", validate="one_to_one")

    snapshot_cols = [
        *ID_COLS,
        *OWN_COLS,
        "SPTC_COUNT",
        "TOTAL_DEPOSIT_BALANCE",
        "TOTAL_DEPOSIT",
        "TRANS_RECORDS",
        "AGE_CLEAN",
        "AGE",
        "CUSTOMER_TENURE_MONTHS",
        "IB_TENURE_MONTHS",
    ]
    available_cols = pq.ParquetFile(NBFO_INPUT_PATH).schema_arrow.names
    snapshot_cols = [col for col in snapshot_cols if col in available_cols]
    snapshot = pd.read_parquet(NBFO_INPUT_PATH, columns=snapshot_cols)
    snapshot["MONTH"] = pd.to_datetime(snapshot["MONTH"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    snapshot = snapshot.drop_duplicates(ID_COLS)

    df = scores.merge(snapshot, on=ID_COLS, how="left", validate="one_to_one")
    df = df.sort_values(["MONTH", "CUSTOMER_NUMBER", "PRODUCT_CODE"]).reset_index(drop=True)
    return df


def enrich_nbfo_features(model_input):
    """Recreate the post-split temporal and affinity features used by train.ipynb."""
    df = model_input.copy()
    month_number = df["MONTH"].dt.month
    split_frames = {
        "train": df[month_number.isin(range(1, 9))].copy(),
        "validation": df[month_number.isin([9])].copy(),
        "test": df[month_number.isin([10])].copy(),
        "excluded": df[~month_number.isin(list(range(1, 11)))].copy(),
    }

    combined = pd.concat(
        [frame.assign(_SPLIT_NAME=split_name) for split_name, frame in split_frames.items()],
        ignore_index=True,
    ).sort_values(["MONTH", "CUSTOMER_NUMBER", "PRODUCT_CODE"]).reset_index(drop=True)

    temporal_base_cols = [
        "ACTIVITY_NAME_NUNIQUE",
        "ACTIVITY_RECORDS",
        "SPTC_COUNT",
        "TOTAL_DEPOSIT_ACCTS",
        "TOTAL_DEPOSIT_BALANCE",
        "TRANS_AMOUNT_SUM",
    ]
    product_count_specs = {
        "CURRENT_ACCOUNT": "COUNT_CA_ACCT",
        "TERM_DEPOSIT": "COUNT_TD_ACCT",
        "CREDIT_CARD": "COUNT_CREDITCARD",
        "DEBIT_CARD": "COUNT_DEBITCARD",
        "LENDING": "COUNT_OF_LOAN",
    }
    activity_count_cols = sorted(
        [
            c
            for c in combined.columns
            if c.startswith("ACTIVITY_TYPE_COUNT_") and not c.endswith(("_SHARE", "_LAST_90D", "_AVG_90D"))
        ]
    )
    rolling_source_cols = [
        "TRANS_ACTIVE_DAYS",
        "ACTIVITY_ACTIVE_DAYS",
        *activity_count_cols,
        *product_count_specs.values(),
        *OWN_COLS,
    ]
    customer_month_cols = list(dict.fromkeys([*temporal_base_cols, *[c for c in rolling_source_cols if c in combined.columns]]))
    customer_month = (
        combined[["CUSTOMER_NUMBER", "MONTH"] + customer_month_cols]
        .drop_duplicates(["CUSTOMER_NUMBER", "MONTH"])
        .sort_values(["CUSTOMER_NUMBER", "MONTH"])
        .reset_index(drop=True)
    )
    grouped = customer_month.groupby("CUSTOMER_NUMBER", sort=False)

    post_split_cols = []
    for col in temporal_base_cols:
        lag = grouped[col].shift(1).fillna(0)
        customer_month[f"{col}_LAG1"] = lag
        customer_month[f"{col}_DIFF1"] = customer_month[col] - lag
        customer_month[f"{col}_ROLL3_MEAN"] = (
            grouped[col]
            .shift(1)
            .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        post_split_cols.extend([f"{col}_LAG1", f"{col}_DIFF1", f"{col}_ROLL3_MEAN"])

    rolling_sum_specs = {
        "ACTIVE_DAYS_LAST_90D": "TRANS_ACTIVE_DAYS",
        "ACTIVITY_COUNT_LAST_90D": "ACTIVITY_RECORDS",
        "ACTIVITY_ACTIVE_DAYS_LAST_90D": "ACTIVITY_ACTIVE_DAYS",
        "TOTAL_TRANSACTION_AMOUNT_90D": "TRANS_AMOUNT_SUM",
    }
    rolling_mean_specs = {"AVG_BALANCE_90D": "TOTAL_DEPOSIT_BALANCE"}
    for new_col, source_col in rolling_sum_specs.items():
        customer_month[new_col] = (
            grouped[source_col]
            .shift(1)
            .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
            .rolling(3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        post_split_cols.append(new_col)

    for new_col, source_col in rolling_mean_specs.items():
        customer_month[new_col] = (
            grouped[source_col]
            .shift(1)
            .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        post_split_cols.append(new_col)

    customer_month["ACTIVITY_RECORDS_PER_ACTIVE_DAY_90D"] = np.where(
        customer_month["ACTIVITY_ACTIVE_DAYS_LAST_90D"] > 0,
        customer_month["ACTIVITY_COUNT_LAST_90D"] / customer_month["ACTIVITY_ACTIVE_DAYS_LAST_90D"],
        0,
    )
    post_split_cols.append("ACTIVITY_RECORDS_PER_ACTIVE_DAY_90D")

    for col in activity_count_cols:
        for suffix, agg in [("LAST_90D", "sum"), ("AVG_90D", "mean")]:
            new_col = f"{col}_{suffix}"
            rolling = (
                grouped[col]
                .shift(1)
                .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
                .rolling(3, min_periods=1)
            )
            customer_month[new_col] = getattr(rolling, agg)().reset_index(level=0, drop=True).fillna(0)
            post_split_cols.append(new_col)

    for product_name, source_col in product_count_specs.items():
        for suffix, agg in [("LAST_90D", "sum"), ("AVG_90D", "mean")]:
            new_col = f"{product_name}_COUNT_{suffix}"
            rolling = (
                grouped[source_col]
                .shift(1)
                .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
                .rolling(3, min_periods=1)
            )
            customer_month[new_col] = getattr(rolling, agg)().reset_index(level=0, drop=True).fillna(0)
            post_split_cols.append(new_col)

    product_open_cols = []
    for product_name, own_col in PRODUCT_TO_OWN.items():
        open_col = f"PRODUCT_OPEN_FLAG_{product_name}"
        product_open_cols.append(open_col)
        customer_month[open_col] = grouped[own_col].diff().gt(0).astype("int8")
    customer_month["PRODUCT_TRANSITION_COUNT_LAST_90D"] = (
        customer_month.groupby("CUSTOMER_NUMBER", sort=False)[product_open_cols]
        .shift(1)
        .groupby(customer_month["CUSTOMER_NUMBER"], sort=False)
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sum(axis=1)
        .fillna(0)
    )
    post_split_cols.append("PRODUCT_TRANSITION_COUNT_LAST_90D")

    combined = combined.merge(
        customer_month[["CUSTOMER_NUMBER", "MONTH"] + post_split_cols],
        on=["CUSTOMER_NUMBER", "MONTH"],
        how="left",
        validate="many_to_one",
    )

    target_conditions = [combined["PRODUCT_NAME"].eq(product_name) for product_name in product_count_specs]
    combined["TARGET_PRODUCT_COUNT_LAST_90D"] = np.select(
        target_conditions,
        [combined[f"{product_name}_COUNT_LAST_90D"] for product_name in product_count_specs],
        default=0,
    )
    combined["TARGET_PRODUCT_COUNT_AVG_90D"] = np.select(
        target_conditions,
        [combined[f"{product_name}_COUNT_AVG_90D"] for product_name in product_count_specs],
        default=0,
    )
    post_split_cols.extend(["TARGET_PRODUCT_COUNT_LAST_90D", "TARGET_PRODUCT_COUNT_AVG_90D"])

    for col in post_split_cols:
        combined[col] = combined[col].fillna(0).astype("float32")

    train_frame = combined.query("_SPLIT_NAME == 'train'").copy()
    co_base = train_frame[["CUSTOMER_NUMBER", "MONTH"] + OWN_COLS].drop_duplicates()
    co_matrix = pd.DataFrame(0.0, index=OWN_COLS, columns=OWN_COLS, dtype="float32")
    for source_col in OWN_COLS:
        source_mask = co_base[source_col] == 1
        for target_col in OWN_COLS:
            value = co_base.loc[source_mask, target_col].mean() if source_mask.any() else 0
            co_matrix.loc[source_col, target_col] = np.float32(value)
    co_matrix = co_matrix.fillna(0).astype("float32")
    base_ownership_rate = co_base[OWN_COLS].mean().to_dict()

    owned_values = combined[OWN_COLS].to_numpy(dtype="float32")
    affinity_sum = np.zeros(len(combined), dtype="float32")
    affinity_max = np.zeros(len(combined), dtype="float32")
    base_rate = np.zeros(len(combined), dtype="float32")
    for product_name, target_own_col in PRODUCT_TO_OWN.items():
        mask = combined["PRODUCT_NAME"].eq(product_name).to_numpy()
        if not mask.any():
            continue
        weights = co_matrix[target_own_col].reindex(OWN_COLS).to_numpy(dtype="float32")
        values = owned_values[mask] * weights
        affinity_sum[mask] = values.sum(axis=1)
        affinity_max[mask] = values.max(axis=1)
        base_rate[mask] = float(base_ownership_rate.get(target_own_col, 0))

    source_count = combined[OWN_COLS].sum(axis=1).astype("float32").to_numpy()
    combined["PRODUCT_AFFINITY_SUM"] = affinity_sum
    combined["PRODUCT_AFFINITY_AVG"] = affinity_sum / np.maximum(source_count, 1)
    combined["PRODUCT_AFFINITY_MAX"] = affinity_max
    combined["PRODUCT_COOCCURRENCE_BASE_RATE"] = base_rate
    combined["PRODUCT_COOCCURRENCE_OWNED_SOURCE_COUNT"] = source_count

    return combined.drop(columns="_SPLIT_NAME").reset_index(drop=True)


def score_nbfo_candidates(candidates):
    scored = candidates[ID_COLS + [TARGET_COL, "propensity_in_h", "propensity_calibrated"] + OWN_COLS].copy()
    scored["RANK"] = scored["propensity_calibrated"].rank(method="first", ascending=False).astype(int)
    scored["ALREADY_OWNED"] = [
        int(row[PRODUCT_TO_OWN[row["PRODUCT_NAME"]]]) for _, row in scored.iterrows()
    ]
    return scored.sort_values("propensity_calibrated", ascending=False)


def render_customer_snapshot(row, mode):
    st.subheader("Customer Snapshot")
    if mode == "nbfo":
        c1, c2, c3 = st.columns(3)
        c1.metric("Products held", int(first_existing(row, ["SPTC_COUNT"], 0)))
        c2.metric("Total balance", format_number(first_existing(row, ["TOTAL_DEPOSIT_BALANCE", "TOTAL_DEPOSIT"], 0)))
        c3.metric("Transactions", format_number(first_existing(row, ["TRANS_RECORDS"], 0)))

        profile = pd.DataFrame(
            [
                ("Customer number", str(row["CUSTOMER_NUMBER"])),
                ("Month", pd.to_datetime(row["MONTH"]).strftime("%Y-%m")),
                ("Age", format_number(first_existing(row, ["AGE_CLEAN", "AGE"], np.nan))),
                ("Customer tenure months", format_number(first_existing(row, ["CUSTOMER_TENURE_MONTHS"], np.nan))),
                ("IB tenure months", format_number(first_existing(row, ["IB_TENURE_MONTHS"], np.nan))),
            ],
            columns=["Field", "Value"],
        )
        products = pd.DataFrame(
            [(product, "Yes" if int(row.get(own_col, 0)) == 1 else "No") for product, own_col in PRODUCT_TO_OWN.items()],
            columns=["Product", "Currently owned"],
        )
        st.dataframe(profile, hide_index=True, use_container_width=True)
        st.dataframe(products, hide_index=True, use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Products held", int(first_existing(row, ["PRODUCT_COUNT"], 0)))
        c2.metric("Financial value", format_number(first_existing(row, ["TOTAL_FINANCIAL_VALUE"], 0)))
        c3.metric("Cluster", int(first_existing(row, ["CLUSTER"], -1)))

        profile = pd.DataFrame(
            [
                ("Customer number", str(row["CUSTOMER_NUMBER"])),
                ("Persona", display_persona_name(row)),
                ("Age group", str(first_existing(row, ["AGE_GROUP"], "-"))),
                ("Sex", str(first_existing(row, ["CLIENT_SEX"], "-"))),
                ("Tenure days", format_number(first_existing(row, ["TENURE_DAYS"], np.nan))),
                ("Dormant", "Yes" if int(first_existing(row, ["IS_DORMANT"], 0)) == 1 else "No"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(profile, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Loading cluster data...")
def load_cluster_data():
    personas = pd.read_parquet(CLUSTER_PERSONA_PATH)
    summary = pd.read_csv(CLUSTER_SUMMARY_PATH)
    try:
        recommendations = pd.read_csv(CLUSTER_RECOMMENDATION_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        recommendations = pd.read_csv(CLUSTER_RECOMMENDATION_PATH, encoding="latin1")
    return personas, summary, recommendations


def select_customer(df, key_prefix, label="Customer"):
    ids = np.sort(df["CUSTOMER_NUMBER"].dropna().astype(int).unique())
    mode = st.radio(
        f"{label} selection",
        ["Random sample", "Choose from list", "Enter ID"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )
    if mode == "Random sample":
        if st.button("Pick random customer", key=f"{key_prefix}_random_button"):
            st.session_state[f"{key_prefix}_random_id"] = int(np.random.choice(ids))
        selected = st.session_state.get(f"{key_prefix}_random_id", int(ids[0]))
        st.caption(f"Selected customer: {selected}")
        return selected
    if mode == "Choose from list":
        sample_size = min(1000, len(ids))
        sampled_ids = np.sort(pd.Series(ids).sample(sample_size, random_state=42).to_numpy())
        return int(st.selectbox("Select customer", sampled_ids, key=f"{key_prefix}_select"))
    return int(st.number_input("Customer number", min_value=int(ids.min()), max_value=int(ids.max()), step=1, key=f"{key_prefix}_input"))


def render_nbfo_tab():
    st.header("IB Product Recommendation")
    st.info("Note: Khách hàng chỉ được recommend các sản phẩm họ chưa sở hữu tại thời điểm dự đoán.")

    nbfo_df = load_nbfo_data()
    available_months = sorted(nbfo_df["MONTH"].dropna().unique())
    selected_month = available_months[-1]

    month_df = nbfo_df[nbfo_df["MONTH"].eq(selected_month)]
    selected_customer = select_customer(month_df, "nbfo", "NBFO customer")
    candidates = month_df[month_df["CUSTOMER_NUMBER"].eq(selected_customer)].copy()

    if candidates.empty:
        st.warning("No eligible product candidates found for this customer/month.")
        return

    scored = score_nbfo_candidates(candidates)
    snapshot = candidates.iloc[0]

    left, right = st.columns([1, 1.35])
    with left:
        render_customer_snapshot(snapshot, "nbfo")

    with right:
        st.subheader("All Eligible Product Propensities")
        propensity_table = scored[
            ["RANK", "PRODUCT_NAME", "PRODUCT_CODE", "propensity_in_h", "propensity_calibrated"]
        ].copy()
        st.dataframe(
            propensity_table.style.format({"propensity_in_h": "{:.4f}", "propensity_calibrated": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            "Mô hình dự đoán khả năng khách hàng sẽ mua sản phẩm tài chính trong 2 tháng tới (h=2) bằng adoption propensity."
        )
        st.markdown(
            """
            <div class="score-note">
              <div><strong>propensity_in_h</strong>: kết quả của predictive modelling, phù hợp với bài toán recommendation. Tuy nhiên còn vấn đề về calibration.</div>
              <div><strong>propensity_calibrated</strong>: propensity score đã qua Platt Scaling nhằm đưa xác suất dự đoán về sát với xác suất thực tế, phù hợp giải quyết các vấn đề khác ngoài ranking và recommendation (downstream,...).</div>
            </div>
            <style>
            .score-note {
                margin: 12px 0 22px 0;
                padding: 12px 14px;
                border-left: 4px solid #2f6f9f;
                background: #f7fafc;
                color: #374151;
                font-size: 15px;
                line-height: 1.55;
            }
            .score-note div + div {
                margin-top: 6px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Recommendation")
        rule_col, value_col = st.columns([1, 1])
        with rule_col:
            rule = st.radio("Strategy", ["Top-K recommend", "Threshold recommend"], horizontal=False)
        with value_col:
            if rule == "Top-K recommend":
                k = st.selectbox("K products", [1, 2, 3], index=2)
                recommended = scored.head(k)
            else:
                threshold = st.slider("Threshold", 0.0, 0.1, 0.01, 0.001, format="%.3f")
                recommended = scored[scored["propensity_calibrated"].ge(threshold)]
                st.markdown(
                    """
                    <div class="threshold-note">
                    Phương pháp threshold được thực hiện với `propensity_calibrated`.
                    Vì xác suất thực tế vốn đã rất thấp do đặc tính imbalance của dữ liệu, nên threshold
                    cũng phải được đặt tương ứng (0-0.1).
                    </div>
                    <style>
                    .threshold-note {
                        margin: 10px 0 14px 0;
                        padding: 10px 12px;
                        background: #fff7ed;
                        border-left: 4px solid #f97316;
                        color: #374151;
                        font-size: 14px;
                        line-height: 1.5;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

        if recommended.empty:
            st.info("No recommendation")
        else:
            st.dataframe(
                recommended[["RANK", "PRODUCT_NAME"]],
                hide_index=True,
                use_container_width=True,
            )


def render_cluster_tab():
    st.header("non-IB Customer Clustering")
    personas, summary, recommendations = load_cluster_data()
    selected_customer = select_customer(personas, "cluster", "Cluster customer")
    row_df = personas[personas["CUSTOMER_NUMBER"].eq(selected_customer)]
    if row_df.empty:
        st.warning("Customer not found in non-IB cluster output.")
        return
    row = row_df.iloc[0]
    cluster_id = int(row["CLUSTER"])

    left, right = st.columns([1, 1.35])
    with left:
        render_customer_snapshot(row, "cluster")

    with right:
        st.subheader("Cluster Profile")
        cluster_summary = summary[summary["CLUSTER"].eq(cluster_id)].copy()
        st.dataframe(cluster_summary, hide_index=True, use_container_width=True)

        st.subheader("Customer vs Cluster Average")
        compare_cols = [
            "AGE",
            "TENURE_DAYS",
            "PRODUCT_COUNT",
            "TOTAL_DEPOSIT",
            "TOTAL_FINANCIAL_VALUE",
            "NET_WORTH_PROXY",
            "SAVINGS_RATE",
            "LOAN_AMOUNT_MEAN",
        ]
        rows = []
        if not cluster_summary.empty:
            avg = cluster_summary.iloc[0]
            for col in compare_cols:
                if col in personas.columns and col in avg.index:
                    rows.append(
                        {
                            "Metric": col,
                            "Customer": row[col],
                            "Cluster avg": avg[col],
                            "Difference": row[col] - avg[col],
                        }
                    )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Persona Campaign Suggestion")
    persona_name = display_persona_name(row)
    campaign = PERSONA_CAMPAIGN.get(
        cluster_id,
        {
            "Chân dung": "Chưa có chân dung được định nghĩa cho persona này.",
            "Hướng tiếp cận": PERSONA_APPROACH.get(
                cluster_id,
                "Chưa có hướng tiếp cận được định nghĩa cho persona này.",
            ),
            "Kênh & hành động đầu tiên": "Chưa có kênh và hành động đầu tiên được định nghĩa.",
        },
    )
    campaign_table = pd.DataFrame(
        [
            {
                "Persona": persona_name,
                "Chân dung": campaign["Chân dung"],
                "Hướng tiếp cận": campaign["Hướng tiếp cận"],
                "Kênh & hành động đầu tiên": campaign["Kênh & hành động đầu tiên"],
            }
        ]
    )
    st.markdown(
        """
        <style>
        .campaign-table {
            width: min(1180px, 100%);
            margin: 10px auto 8px auto;
        }
        .campaign-table table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 16px;
            line-height: 1.55;
            background: #ffffff;
        }
        .campaign-table th {
            background: #f6f8fb;
            color: #374151;
            font-weight: 700;
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        .campaign-table td {
            padding: 18px;
            border: 1px solid #e5e7eb;
            vertical-align: top;
            text-align: left;
            word-break: normal;
            overflow-wrap: anywhere;
        }
        .campaign-table th:nth-child(1),
        .campaign-table td:nth-child(1) {
            width: 20%;
            font-weight: 650;
        }
        .campaign-table th:nth-child(2),
        .campaign-table td:nth-child(2) {
            width: 27%;
        }
        .campaign-table th:nth-child(3),
        .campaign-table td:nth-child(3) {
            width: 28%;
        }
        .campaign-table th:nth-child(4),
        .campaign-table td:nth-child(4) {
            width: 25%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="campaign-table">{campaign_table.to_html(index=False, escape=True)}</div>',
        unsafe_allow_html=True,
    )


def main():
    st.title("GCON Demo Dashboard")
    st.caption("Interactive demo for IB product recommendation and non-IB customer clustering.")
    nbfo_tab, cluster_tab = st.tabs(["NBFO Recommendation", "non-IB Clustering"])
    with nbfo_tab:
        render_nbfo_tab()
    with cluster_tab:
        render_cluster_tab()


def is_running_inside_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def find_free_port(start_port=8501, max_tries=20):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found from {start_port} to {start_port + max_tries - 1}")


def launch_streamlit_app():
    port = find_free_port()
    url = f"http://localhost:{port}"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    print(f"Starting GCON Demo Dashboard at {url}")
    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    time.sleep(2)
    webbrowser.open(url)
    process.wait()


if __name__ == "__main__":
    if is_running_inside_streamlit():
        main()
    else:
        launch_streamlit_app()
