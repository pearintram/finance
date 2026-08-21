import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import brentq

# =====================================================================
# CẤU HÌNH TRANG
# =====================================================================
st.set_page_config(page_title="Mô phỏng Đầu tư Định kỳ", page_icon="📈", layout="wide")

# =====================================================================
# THAM SỐ MẶC ĐỊNH CHO CÁC KÊNH ĐẦU TƯ (lợi nhuận & biến động theo năm)
# Đây là giả định minh hoạ, người dùng có thể tự chỉnh trong app
# =====================================================================
CHANNELS_DEFAULT = {
    "Tiết kiệm ngân hàng": {"mean": 0.06, "vol": 0.01, "mota": "Gần như không rủi ro, lãi suất thấp"},
    "Trái phiếu": {"mean": 0.08, "vol": 0.04, "mota": "Rủi ro thấp, lợi nhuận ổn định"},
    "Vàng": {"mean": 0.09, "vol": 0.15, "mota": "Biến động trung bình, mang tính trú ẩn"},
    "Bất động sản": {"mean": 0.12, "vol": 0.18, "mota": "Thanh khoản thấp, biến động theo chu kỳ"},
    "Chứng khoán (VN-Index)": {"mean": 0.14, "vol": 0.25, "mota": "Biến động cao, lợi nhuận kỳ vọng cao"},
    "Crypto": {"mean": 0.30, "vol": 0.80, "mota": "⚠️ Biến động cực cao, rủi ro rất lớn"},
}

VOL_MULTIPLIER = {"Thấp": 0.7, "Trung bình (mặc định)": 1.0, "Cao": 1.4}
MAU_KENH = {
    "Tiết kiệm ngân hàng": "#3498DB",
    "Trái phiếu": "#1ABC9C",
    "Vàng": "#F1C40F",
    "Bất động sản": "#9B59B6",
    "Chứng khoán (VN-Index)": "#E74C3C",
    "Crypto": "#34495E",
    "Danh mục hỗn hợp": "#2ECC71",
}


def dinh_dang_tien(so_trieu):
    """Định dạng số tiền (đơn vị triệu VNĐ) sang chuỗi dễ đọc."""
    if abs(so_trieu) >= 1000:
        return f"{so_trieu / 1000:,.2f} tỷ"
    return f"{so_trieu:,.1f} triệu"


# =====================================================================
# LÕI MÔ PHỎNG
# =====================================================================
@st.cache_data(show_spinner=False)
def sinh_loi_nhuan_hang_thang(months, mean_annual, vol_annual, dist, n_sims, seed=42):
    """Sinh ma trận lợi nhuận hàng tháng ngẫu nhiên (n_sims x months)."""
    rng = np.random.default_rng(seed)
    mean_m = (1 + mean_annual) ** (1 / 12) - 1
    vol_m = vol_annual / np.sqrt(12)
    if dist == "Đuôi dày (biến cố cực đoan dễ xảy ra hơn)":
        df = 4
        raw = rng.standard_t(df, size=(n_sims, months))
        raw = raw / raw.std() * vol_m
        draws = mean_m + raw
    else:
        draws = rng.normal(mean_m, vol_m, size=(n_sims, months))
    return draws


def ap_dung_cu_soc(draws, shock_month_idx, shock_pct):
    """Ghi đè lợi nhuận của một tháng cụ thể bằng cú sốc giả định (áp dụng cho mọi kịch bản)."""
    draws = draws.copy()
    if shock_month_idx is not None and 0 <= shock_month_idx < draws.shape[1]:
        draws[:, shock_month_idx] = shock_pct
    return draws


def tinh_gia_tri_tich_luy(monthly_invest, draws):
    """Tính giá trị tài sản tích luỹ theo từng tháng cho mỗi kịch bản mô phỏng."""
    n_sims, months = draws.shape
    values = np.zeros((n_sims, months + 1))
    for m in range(months):
        values[:, m + 1] = (values[:, m] + monthly_invest) * (1 + draws[:, m])
    return values


def tinh_irr_nam(monthly_invest, months, final_value):
    """Tính lãi suất nội bộ (IRR) quy đổi theo năm cho dòng tiền góp đều mỗi tháng."""
    def npv(r):
        total = sum(-monthly_invest / (1 + r) ** t for t in range(months))
        total += final_value / (1 + r) ** months
        return total
    try:
        r = brentq(npv, -0.99, 50)
    except ValueError:
        return None
    return (1 + r) ** 12 - 1


def xac_suat_lo_theo_nam(values, monthly_invest, years):
    """Tính xác suất giá trị tài sản thấp hơn số tiền đã góp, tại mốc cuối mỗi năm."""
    ket_qua = []
    for y in range(1, years + 1):
        idx = y * 12
        invested = monthly_invest * idx
        prob = np.mean(values[:, idx] < invested)
        ket_qua.append({"Năm": y, "Xác suất lỗ (%)": prob * 100})
    return pd.DataFrame(ket_qua)


# =====================================================================
# GIAO DIỆN - SIDEBAR (DỮ LIỆU VÀO)
# =====================================================================
st.title("📈 Mô phỏng Đầu tư Định kỳ")
st.markdown(
    "Ứng dụng mô phỏng kết quả đầu tư đều đặn hàng tháng vào nhiều kênh khác nhau, "
    "có tính đến **biến động thị trường thực tế** thay vì lãi suất cố định trơn tru. "
    "Dùng để tham khảo, so sánh kịch bản và hiểu rủi ro — **không phải lời khuyên đầu tư chính thức**."
)

st.sidebar.header("⚙️ Thiết lập mô phỏng")

so_tien_thang = st.sidebar.number_input(
    "Số tiền đầu tư mỗi tháng (triệu VNĐ)", min_value=1, max_value=1000, value=10, step=1
)
so_nam = st.sidebar.slider("Thời gian đầu tư (năm)", min_value=1, max_value=30, value=10)
so_thang = so_nam * 12

kenh_chon = st.sidebar.multiselect(
    "Chọn kênh đầu tư để so sánh",
    options=list(CHANNELS_DEFAULT.keys()),
    default=["Tiết kiệm ngân hàng", "Vàng", "Chứng khoán (VN-Index)"],
)

muc_bien_dong = st.sidebar.select_slider(
    "Mức độ biến động giả định", options=list(VOL_MULTIPLIER.keys()), value="Trung bình (mặc định)"
)
he_so_bien_dong = VOL_MULTIPLIER[muc_bien_dong]

loai_phan_phoi = st.sidebar.radio(
    "Kiểu biến động lợi nhuận",
    options=["Chuẩn (Normal)", "Đuôi dày (biến cố cực đoan dễ xảy ra hơn)"],
    help="'Đuôi dày' mô phỏng thực tế hơn: các cú tăng/giảm mạnh bất thường xảy ra thường xuyên hơn phân phối chuẩn.",
)

so_lan_mo_phong = st.sidebar.slider(
    "Số lần mô phỏng Monte Carlo", min_value=100, max_value=3000, value=800, step=100
)

with st.sidebar.expander("✏️ Tuỳ chỉnh giả định lợi nhuận / biến động"):
    st.caption("Có thể chỉnh lại lợi nhuận kỳ vọng và độ biến động (%/năm) cho từng kênh.")
    gia_dinh_tuy_chinh = {}
    for ten in kenh_chon:
        mac_dinh = CHANNELS_DEFAULT[ten]
        col1, col2 = st.columns(2)
        with col1:
            mean_pct = st.number_input(
                f"LN kỳ vọng - {ten} (%/năm)", value=float(mac_dinh["mean"] * 100), key=f"mean_{ten}"
            )
        with col2:
            vol_pct = st.number_input(
                f"Biến động - {ten} (%/năm)", value=float(mac_dinh["vol"] * 100), key=f"vol_{ten}"
            )
        gia_dinh_tuy_chinh[ten] = {"mean": mean_pct / 100, "vol": (vol_pct / 100) * he_so_bien_dong}

with st.sidebar.expander("💥 Thêm cú sốc thị trường (tuỳ chọn)"):
    bat_cu_soc = st.checkbox("Kích hoạt cú sốc giả định", value=False)
    nam_cu_soc = st.slider("Cú sốc xảy ra vào năm thứ", 1, so_nam, min(3, so_nam), disabled=not bat_cu_soc)
    muc_cu_soc = st.slider("Mức giảm/tăng của cú sốc (%)", -80, 50, -30, disabled=not bat_cu_soc)
    kenh_bi_soc = st.multiselect(
        "Áp dụng cú sốc cho kênh nào", options=kenh_chon, default=kenh_chon, disabled=not bat_cu_soc
    )

with st.sidebar.expander("🧺 Mô phỏng danh mục hỗn hợp (tuỳ chọn)"):
    bat_danh_muc = st.checkbox("Bật danh mục hỗn hợp (chia vốn cho nhiều kênh)", value=False)
    trong_so = {}
    if bat_danh_muc and kenh_chon:
        st.caption("Nhập tỷ trọng (%) cho từng kênh, tổng nên bằng 100%.")
        for ten in kenh_chon:
            trong_so[ten] = st.slider(f"Tỷ trọng - {ten} (%)", 0, 100, int(100 / len(kenh_chon)), key=f"w_{ten}")
        tong_ts = sum(trong_so.values())
        if tong_ts == 0:
            st.warning("Tổng tỷ trọng đang bằng 0%, vui lòng điều chỉnh.")
        else:
            st.caption(f"Tổng tỷ trọng hiện tại: {tong_ts}%")

if not kenh_chon:
    st.warning("⬅️ Vui lòng chọn ít nhất một kênh đầu tư ở thanh bên trái để bắt đầu.")
    st.stop()

# =====================================================================
# CHẠY MÔ PHỎNG CHO TỪNG KÊNH
# =====================================================================
shock_month_idx = (nam_cu_soc - 1) * 12 + 5 if bat_cu_soc else None  # áp cú sốc vào giữa năm được chọn

ket_qua_kenh = {}
draws_theo_kenh = {}
for ten in kenh_chon:
    tham_so = gia_dinh_tuy_chinh[ten]
    draws = sinh_loi_nhuan_hang_thang(so_thang, tham_so["mean"], tham_so["vol"], loai_phan_phoi, so_lan_mo_phong)
    if bat_cu_soc and ten in kenh_bi_soc:
        draws = ap_dung_cu_soc(draws, shock_month_idx, muc_cu_soc / 100)
    draws_theo_kenh[ten] = draws
    values = tinh_gia_tri_tich_luy(so_tien_thang, draws)
    ket_qua_kenh[ten] = values

# Danh mục hỗn hợp: cộng lợi nhuận theo tỷ trọng rồi mô phỏng lại
if bat_danh_muc and kenh_chon and sum(trong_so.values()) > 0:
    tong_ts = sum(trong_so.values())
    portfolio_draws = np.zeros_like(draws_theo_kenh[kenh_chon[0]])
    for ten in kenh_chon:
        portfolio_draws += draws_theo_kenh[ten] * (trong_so[ten] / tong_ts)
    portfolio_values = tinh_gia_tri_tich_luy(so_tien_thang, portfolio_draws)
    ket_qua_kenh["Danh mục hỗn hợp"] = portfolio_values

invested_total = so_tien_thang * so_thang

# =====================================================================
# BẢNG TÓM TẮT KẾT QUẢ
# =====================================================================
st.subheader("📊 Kết quả tổng quan")

bang_tom_tat = []
for ten, values in ket_qua_kenh.items():
    final = values[:, -1]
    p10, p50, p90 = np.percentile(final, [10, 50, 90])
    irr = tinh_irr_nam(so_tien_thang, so_thang, p50)
    prob_lo = np.mean(final < invested_total) * 100
    bang_tom_tat.append({
        "Kênh": ten,
        "Vốn đã góp (triệu)": invested_total,
        "Giá trị trung vị (triệu)": round(p50, 1),
        "Kịch bản xấu - P10 (triệu)": round(p10, 1),
        "Kịch bản tốt - P90 (triệu)": round(p90, 1),
        "IRR ước tính (%/năm)": round(irr * 100, 2) if irr is not None else None,
        "Xác suất lỗ cuối kỳ (%)": round(prob_lo, 1),
    })
df_tom_tat = pd.DataFrame(bang_tom_tat).set_index("Kênh")
st.dataframe(df_tom_tat, use_container_width=True)

st.caption(
    f"Tổng số tiền đã đầu tư sau {so_nam} năm: **{dinh_dang_tien(invested_total)} VNĐ** "
    f"(góp {so_tien_thang} triệu/tháng)."
)

# =====================================================================
# BIỂU ĐỒ 1: DẢI PERCENTILE (FAN CHART) THEO TỪNG KÊNH
# =====================================================================
st.subheader("📈 Tăng trưởng tài sản theo thời gian (dải kịch bản 10% - 90%)")

thang_truc = np.arange(0, so_thang + 1)
nam_truc = thang_truc / 12

fig1, ax1 = plt.subplots(figsize=(11, 5))
for ten, values in ket_qua_kenh.items():
    mau = MAU_KENH.get(ten, None)
    p10 = np.percentile(values, 10, axis=0)
    p50 = np.percentile(values, 50, axis=0)
    p90 = np.percentile(values, 90, axis=0)
    ax1.plot(nam_truc, p50, label=f"{ten} (trung vị)", color=mau, linewidth=2)
    ax1.fill_between(nam_truc, p10, p90, color=mau, alpha=0.15)

von_da_gop = so_tien_thang * thang_truc
ax1.plot(nam_truc, von_da_gop, linestyle="--", color="gray", linewidth=1.5, label="Vốn đã góp (gốc)")

if bat_cu_soc:
    ax1.axvline(x=nam_cu_soc, color="red", linestyle=":", linewidth=1.5)
    ax1.text(nam_cu_soc, ax1.get_ylim()[1] * 0.95, " Cú sốc", color="red", fontsize=9)

ax1.set_xlabel("Số năm")
ax1.set_ylabel("Giá trị tài sản (triệu VNĐ)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)
st.caption("Vùng tô màu thể hiện khoảng kết quả có thể xảy ra (từ kịch bản xấu P10 đến kịch bản tốt P90).")

# =====================================================================
# BIỂU ĐỒ 2: VÙNG RỦI RO - XÁC SUẤT LỖ THEO THỜI GIAN
# =====================================================================
st.subheader("⚠️ Xác suất bị lỗ tại các mốc thời gian")

fig2, ax2 = plt.subplots(figsize=(11, 4))
for ten, values in ket_qua_kenh.items():
    mau = MAU_KENH.get(ten, None)
    df_risk = xac_suat_lo_theo_nam(values, so_tien_thang, so_nam)
    ax2.plot(df_risk["Năm"], df_risk["Xác suất lỗ (%)"], marker="o", label=ten, color=mau)

ax2.set_xlabel("Số năm nắm giữ")
ax2.set_ylabel("Xác suất lỗ (%)")
ax2.set_ylim(0, 100)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
st.caption("Xác suất lỗ = tỷ lệ kịch bản mô phỏng mà giá trị tài sản thấp hơn tổng số vốn đã góp tại thời điểm đó.")

# =====================================================================
# BIỂU ĐỒ 3: BA KỊCH BẢN LẠC QUAN / TRUNG BÌNH / BI QUAN (CHO 1 KÊNH)
# =====================================================================
st.subheader("🎭 So sánh kịch bản Lạc quan / Trung bình / Bi quan")

kenh_chi_tiet = st.selectbox("Chọn kênh để xem chi tiết 3 kịch bản", options=kenh_chon)
ts = gia_dinh_tuy_chinh[kenh_chi_tiet]

def duong_deu(mean_annual):
    mean_m = (1 + mean_annual) ** (1 / 12) - 1
    gia_tri = np.zeros(so_thang + 1)
    for m in range(so_thang):
        gia_tri[m + 1] = (gia_tri[m] + so_tien_thang) * (1 + mean_m)
    return gia_tri

lac_quan = duong_deu(ts["mean"] + ts["vol"])
trung_binh = duong_deu(ts["mean"])
bi_quan = duong_deu(max(ts["mean"] - ts["vol"], -0.9))

fig3, ax3 = plt.subplots(figsize=(11, 4))
ax3.plot(nam_truc, lac_quan, color="#2ECC71", label="Lạc quan (LN kỳ vọng + biến động)")
ax3.plot(nam_truc, trung_binh, color="#3498DB", label="Trung bình (LN kỳ vọng)")
ax3.plot(nam_truc, bi_quan, color="#E74C3C", label="Bi quan (LN kỳ vọng - biến động)")
ax3.plot(nam_truc, von_da_gop, linestyle="--", color="gray", label="Vốn đã góp (gốc)")
ax3.set_xlabel("Số năm")
ax3.set_ylabel("Giá trị tài sản (triệu VNĐ)")
ax3.set_title(f"Kênh: {kenh_chi_tiet}")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
st.caption(
    "Ba đường trên là kịch bản tăng trưởng **đều đặn giả định** (không có biến động ngẫu nhiên theo tháng), "
    "dùng để hình dung nhanh khoảng kết quả tốt/xấu, khác với dải mô phỏng Monte Carlo ở biểu đồ phía trên."
)

# =====================================================================
# PHẦN TƯ VẤN / INSIGHT
# =====================================================================
st.subheader("💡 Nhận định dựa trên kết quả mô phỏng")

goi_y = []
for ten, values in ket_qua_kenh.items():
    final = values[:, -1]
    prob_lo_cuoi = np.mean(final < invested_total) * 100
    idx_1y = min(12, so_thang)
    prob_lo_1y = np.mean(values[:, idx_1y] < so_tien_thang * idx_1y) * 100
    p50 = np.percentile(final, 50)
    goi_y.append((ten, prob_lo_1y, prob_lo_cuoi, p50))

for ten, prob_1y, prob_cuoi, p50 in goi_y:
    st.markdown(
        f"- **{ten}**: nếu rút sau **1 năm**, xác suất lỗ khoảng **{prob_1y:.0f}%**; "
        f"nếu giữ đủ **{so_nam} năm**, xác suất lỗ giảm còn khoảng **{prob_cuoi:.0f}%**, "
        f"giá trị trung vị ước tính **{dinh_dang_tien(p50)} VNĐ**."
    )

kenh_an_toan_nhat = min(goi_y, key=lambda x: x[2])[0]
kenh_loi_nhuan_cao_nhat = max(goi_y, key=lambda x: x[3])[0]

st.markdown(
    f"""
**Nhận định chung:**
- Kênh có xác suất lỗ thấp nhất khi giữ đủ kỳ hạn: **{kenh_an_toan_nhat}**.
- Kênh có giá trị trung vị kỳ vọng cao nhất: **{kenh_loi_nhuan_cao_nhat}**.
- Nhìn chung, các kênh biến động cao thường có xác suất lỗ cao hơn đáng kể trong ngắn hạn (dưới 3 năm),
  nhưng xác suất này thường giảm dần khi kéo dài thời gian đầu tư nhờ hiệu ứng trung bình hoá chi phí (DCA).
- Thay vì dồn toàn bộ vào một kênh, việc **chia vốn cho nhiều kênh có mức rủi ro khác nhau** (dùng tính năng
  "Danh mục hỗn hợp" ở thanh bên trái) thường giúp giảm biến động tổng thể của danh mục.
"""
)

# =====================================================================
# DISCLAIMER
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:13px; color:gray;'>"
    "⚠️ <b>Lưu ý:</b> Đây là công cụ mô phỏng mang tính chất minh hoạ và giáo dục, dựa trên các giả định "
    "về lợi nhuận và biến động do người dùng nhập/chỉnh. Kết quả mô phỏng "
    "<b>không đại diện cho hiệu suất thực tế trong tương lai</b> và "
    "<b>không phải là lời khuyên đầu tư hay tư vấn tài chính chính thức</b>. "
    "Vui lòng tham khảo ý kiến chuyên gia tài chính được cấp phép trước khi ra quyết định đầu tư."
    "</p>",
    unsafe_allow_html=True,
)
