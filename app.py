from __future__ import annotations

import io
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predictor import load_assets, predict_frame

st.set_page_config(
    page_title="Backorder | Hỗ trợ chuỗi cung ứng",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Hệ thống hỗ trợ dự đoán nguy cơ thiếu hàng")
st.write(
    "Ứng dụng sử dụng mô hình Random Forest – Luồng A (Feature Engineering) "
    "để ước lượng nguy cơ một sản phẩm rơi vào tình trạng backorder."
)
st.caption(
    "Đây là công cụ hỗ trợ sàng lọc và ra quyết định. Kết quả dự đoán không thay thế "
    "việc kiểm tra tồn kho thực tế, lịch giao hàng và xác nhận từ nhà cung cấp."
)

@st.cache_resource(show_spinner="Đang tải mô hình và bộ tiền xử lý...")
def get_assets():
    return load_assets()

try:
    assets = get_assets()
except Exception as exc:
    st.error(f"Không thể tải mô hình. Kiểm tra thư mục models. Chi tiết: {exc}")
    st.stop()

metadata = assets["metadata"]
feature_cols = metadata["feature_cols"]
threshold_presets = {
    "Cân bằng Precision và Recall (F1 tối đa trên Validation)": float(metadata["thresholds"]["f1_best"]),
    "Ưu tiên phát hiện thiếu hàng (Recall ≥ 80% trên Validation)": float(metadata["thresholds"]["recall_80"]),
    "Ngưỡng mặc định 0,50": 0.5,
}

with st.sidebar:
    st.header("Thiết lập cảnh báo")
    threshold_label = st.selectbox("Chiến lược ngưỡng", list(threshold_presets.keys()))
    threshold = threshold_presets[threshold_label]
    st.metric("Ngưỡng phân loại đang dùng", f"{threshold:.9f}")
    st.caption(
        "Các ngưỡng F1 và Recall được khảo sát từ tập Validation của bộ dữ liệu hiện tại. "
        "Có thể điều chỉnh theo chi phí cảnh báo sai và chi phí bỏ sót thiếu hàng."
    )
    st.divider()
    st.subheader("Thông tin mô hình")
    st.write("**Mô hình:** Random Forest")
    st.write("**Luồng:** A – Feature Engineering")
    st.write("**Số thuộc tính đầu vào:**", len(feature_cols))

single_tab, batch_tab, about_tab = st.tabs([
    "Dự đoán một sản phẩm", "Dự đoán hàng loạt bằng CSV", "Thông tin ứng dụng"
])

# ---------------------------- Dự đoán một sản phẩm ----------------------------
with single_tab:
    st.subheader("Nhập thông tin sản phẩm")
    st.write("Nhập các thuộc tính tồn kho, vận chuyển, dự báo và rủi ro nhà cung cấp.")

    default_values = metadata.get("default_values", {})

    def default_value(col: str, fallback: float = 0.0) -> float:
        try:
            value = float(default_values.get(col, fallback))
            return value if pd.notna(value) else fallback
        except (TypeError, ValueError):
            return fallback

    with st.form("single_prediction_form"):
        sku = st.text_input("Mã sản phẩm (SKU)", value="SKU-DEMO-001")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Tồn kho và vận chuyển**")
            national_inv = st.number_input("Tồn kho quốc gia", value=default_value("national_inv"), step=1.0)
            lead_time = st.number_input("Thời gian cung ứng (lead_time)", value=default_value("lead_time"), min_value=0.0, step=1.0)
            in_transit_qty = st.number_input("Số lượng hàng đang vận chuyển", value=default_value("in_transit_qty"), step=1.0)
            min_bank = st.number_input("Mức tồn kho tối thiểu", value=default_value("min_bank"), step=1.0)
            pieces_past_due = st.number_input("Số lượng hàng quá hạn", value=default_value("pieces_past_due"), step=1.0)
            local_bo_qty = st.number_input("Số lượng backorder tại địa phương", value=default_value("local_bo_qty"), step=1.0)

        with c2:
            st.markdown("**Dự báo và doanh số**")
            forecast_3_month = st.number_input("Dự báo nhu cầu 3 tháng", value=default_value("forecast_3_month"), step=1.0)
            forecast_6_month = st.number_input("Dự báo nhu cầu 6 tháng", value=default_value("forecast_6_month"), step=1.0)
            forecast_9_month = st.number_input("Dự báo nhu cầu 9 tháng", value=default_value("forecast_9_month"), step=1.0)
            sales_1_month = st.number_input("Doanh số 1 tháng", value=default_value("sales_1_month"), step=1.0)
            sales_3_month = st.number_input("Doanh số 3 tháng", value=default_value("sales_3_month"), step=1.0)
            sales_6_month = st.number_input("Doanh số 6 tháng", value=default_value("sales_6_month"), step=1.0)
            sales_9_month = st.number_input("Doanh số 9 tháng", value=default_value("sales_9_month"), step=1.0)

        with c3:
            st.markdown("**Hiệu suất và chỉ báo rủi ro**")
            perf_6_month_avg = st.number_input("Hiệu suất nhà cung cấp 6 tháng", value=default_value("perf_6_month_avg"), format="%.4f")
            perf_12_month_avg = st.number_input("Hiệu suất nhà cung cấp 12 tháng", value=default_value("perf_12_month_avg"), format="%.4f")
            potential_issue = st.selectbox("Có vấn đề tiềm ẩn?", [0, 1], format_func=lambda x: "Có" if x else "Không")
            deck_risk = st.selectbox("Deck risk", [0, 1], format_func=lambda x: "Có" if x else "Không")
            oe_constraint = st.selectbox("Ràng buộc OE", [0, 1], format_func=lambda x: "Có" if x else "Không")
            ppap_risk = st.selectbox("PPAP risk", [0, 1], format_func=lambda x: "Có" if x else "Không")
            stop_auto_buy = st.selectbox("Dừng tự động mua hàng?", [0, 1], format_func=lambda x: "Có" if x else "Không")
            rev_stop = st.selectbox("Rev stop?", [0, 1], format_func=lambda x: "Có" if x else "Không")

        submitted = st.form_submit_button("Dự đoán nguy cơ thiếu hàng", type="primary", use_container_width=True)

    if submitted:
        input_row = {
            "sku": sku.strip() or "SKU-DEMO-001",
            "national_inv": national_inv,
            "lead_time": lead_time,
            "in_transit_qty": in_transit_qty,
            "forecast_3_month": forecast_3_month,
            "forecast_6_month": forecast_6_month,
            "forecast_9_month": forecast_9_month,
            "sales_1_month": sales_1_month,
            "sales_3_month": sales_3_month,
            "sales_6_month": sales_6_month,
            "sales_9_month": sales_9_month,
            "min_bank": min_bank,
            "potential_issue": potential_issue,
            "pieces_past_due": pieces_past_due,
            "perf_6_month_avg": perf_6_month_avg,
            "perf_12_month_avg": perf_12_month_avg,
            "local_bo_qty": local_bo_qty,
            "deck_risk": deck_risk,
            "oe_constraint": oe_constraint,
            "ppap_risk": ppap_risk,
            "stop_auto_buy": stop_auto_buy,
            "rev_stop": rev_stop,
        }
        try:
            result, warnings_list = predict_frame(pd.DataFrame([input_row]), assets, threshold)
            row = result.iloc[0]
            probability = float(row["xac_suat_thieu_hang"])
            st.divider()
            st.subheader("Kết quả dự đoán")
            k1, k2, k3 = st.columns(3)
            k1.metric("Xác suất thiếu hàng", f"{probability:.1%}")
            k2.metric("Ngưỡng cảnh báo", f"{threshold:.9f}")
            st.caption(f"Chiến lược đang dùng: {threshold_label}")
            k3.metric("Kết luận", "Cần rà soát" if int(row["du_doan"]) else "Chưa cảnh báo")
            if int(row["du_doan"]):
                st.error("Sản phẩm vượt ngưỡng cảnh báo nguy cơ thiếu hàng.")
            else:
                st.success("Sản phẩm chưa vượt ngưỡng cảnh báo tại cấu hình hiện tại.")
            st.info(row["goi_y_xu_ly"])
            for warning in warnings_list:
                st.warning(warning)
        except Exception as exc:
            st.error(f"Không thể thực hiện dự đoán: {exc}")

# ---------------------------- Dự đoán hàng loạt ----------------------------
with batch_tab:
    st.subheader("Tải CSV để dự đoán nhiều sản phẩm")
    st.write(
        "CSV cần có đủ 21 thuộc tính đầu vào như `test.csv`. Có thể kèm cột `sku`; "
        "không cần cột `went_on_backorder`."
    )
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], key="batch_csv")
    if uploaded_file is not None:
        try:
            raw_csv = uploaded_file.getvalue()
            upload_fingerprint = hashlib.sha256(raw_csv).hexdigest()

            # Xóa kết quả cũ khi người dùng tải lên một file CSV khác.
            if st.session_state.get("batch_upload_fingerprint") != upload_fingerprint:
                for state_key in ("batch_result", "batch_warnings", "batch_threshold", "batch_threshold_label", "batch_input_df"):
                    st.session_state.pop(state_key, None)
                st.session_state["batch_upload_fingerprint"] = upload_fingerprint

            input_df = pd.read_csv(io.BytesIO(raw_csv), low_memory=False)

            # Một số bản CSV phát hành kèm dòng tổng kết ở cuối, ví dụ
            # "(242075 rows)" trong cột sku. Đây không phải sản phẩm thực.
            summary_rows_removed = 0
            id_col = metadata.get("id_col", "sku")
            if id_col in input_df.columns:
                summary_mask = input_df[id_col].astype("string").str.strip().str.match(
                    r"^\(?\s*\d+\s+rows?\s*\)?$", case=False, na=False
                )
                summary_rows_removed = int(summary_mask.sum())
                if summary_rows_removed:
                    input_df = input_df.loc[~summary_mask].copy()
                    st.info(
                        f"Đã loại {summary_rows_removed} dòng tổng kết ở cuối CSV; "
                        "các dòng này không phải dữ liệu sản phẩm."
                    )

            # Lưu dữ liệu đầu vào để có thể tính lại ngay khi người dùng đổi ngưỡng.
            st.session_state["batch_input_df"] = input_df.copy()
            st.write("Kích thước dữ liệu hợp lệ:", input_df.shape)
            st.dataframe(input_df.head(10), use_container_width=True)
            if st.button("Chạy dự đoán cho file CSV", type="primary"):
                result, warnings_list = predict_frame(input_df, assets, threshold)
                result["chien_luoc_nguong"] = threshold_label
                st.session_state["batch_result"] = result
                st.session_state["batch_warnings"] = warnings_list
                st.session_state["batch_threshold"] = float(threshold)
                st.session_state["batch_threshold_label"] = threshold_label
        except Exception as exc:
            st.error(f"Không đọc được CSV: {exc}")

    if "batch_result" in st.session_state:
        # Sidebar thay đổi ngưỡng sẽ làm Streamlit chạy lại script. Tự động
        # tính lại kết quả batch để số cảnh báo, nhãn và ngưỡng luôn nhất quán.
        previous_threshold = st.session_state.get("batch_threshold")
        if previous_threshold is not None and not abs(float(previous_threshold) - float(threshold)) < 1e-12:
            batch_input = st.session_state.get("batch_input_df")
            if batch_input is not None:
                result, warnings_list = predict_frame(batch_input, assets, threshold)
                result["chien_luoc_nguong"] = threshold_label
                st.session_state["batch_result"] = result
                st.session_state["batch_warnings"] = warnings_list
                st.session_state["batch_threshold"] = float(threshold)
                st.session_state["batch_threshold_label"] = threshold_label
                st.info("Ngưỡng cảnh báo đã thay đổi; kết quả dự đoán hàng loạt đã được tự động cập nhật theo ngưỡng mới.")

        result = st.session_state["batch_result"]
        positives = int(result["du_doan"].sum())
        total = len(result)
        avg_probability = float(result["xac_suat_thieu_hang"].mean()) if total else 0.0
        st.caption(
            "Ngưỡng đang áp dụng cho bảng kết quả: "
            f"{float(st.session_state.get('batch_threshold', threshold)):.9f} "
            f"({st.session_state.get('batch_threshold_label', threshold_label)})"
        )
        a, b, c = st.columns(3)
        a.metric("Tổng sản phẩm", f"{total:,}")
        b.metric("Sản phẩm cần rà soát", f"{positives:,}")
        c.metric("Xác suất trung bình", f"{avg_probability:.1%}")
        st.dataframe(
            result.sort_values("xac_suat_thieu_hang", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "xac_suat_thieu_hang": st.column_config.NumberColumn(
                    "Xác suất thiếu hàng", format="%.9f"
                ),
                "nguong_ap_dung": st.column_config.NumberColumn(
                    "Ngưỡng áp dụng", format="%.9f"
                ),
            },
        )
        csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Tải kết quả dự đoán CSV",
            data=csv_bytes,
            file_name="ket_qua_du_doan_backorder.csv",
            mime="text/csv",
        )
        for warning in st.session_state.get("batch_warnings", []):
            st.warning(warning)

# ---------------------------- Thông tin và giới hạn ----------------------------
with about_tab:
    st.subheader("Quy trình xử lý")
    st.markdown(
        """
        1. Nhận dữ liệu đầu vào của sản phẩm.
        2. Xử lý mã thiếu đặc biệt `-99` và điền dữ liệu thiếu bằng trung vị đã học từ tập train.
        3. Tạo 6 đặc trưng bổ sung theo Luồng A – Feature Engineering.
        4. Chuẩn hóa bằng RobustScaler đã fit trên tập train.
        5. Dùng Random Forest để ước lượng xác suất `went_on_backorder`.
        6. So sánh xác suất với ngưỡng cảnh báo để tạo kết luận và gợi ý rà soát.
        """
    )
    st.subheader("Giới hạn của phiên bản hiện tại")
    st.markdown(
        """
        - Mô hình dự đoán nguy cơ backorder, chưa tính toán lượng đặt hàng tối ưu.
        - Ứng dụng chưa kết nối dữ liệu tồn kho hoặc nhà cung cấp theo thời gian thực.
        - Gợi ý xử lý là quy tắc hỗ trợ rà soát, không phải lệnh đặt hàng tự động.
        - Ngưỡng cảnh báo được khảo sát trên tập Validation hiện có; cần đánh giá lại khi áp dụng dữ liệu thực tế.
        """
    )
    st.caption("Đề tài 10 – Tối ưu hóa chuỗi cung ứng | Học phần Nhập môn học máy")
