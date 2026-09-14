"""
app.py
------
Streamlit UI - Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt
bằng Google OR-Tools + Guided Local Search (GLS), dùng OSRM/OSM làm mạng
lưới đường thực tế.

PHẠM VI NGHIÊN CỨU: Tuyến Lê Văn Việt và khu vực lân cận, TP. Thủ Đức, TP.HCM.
"""

from __future__ import annotations

import io
import time

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

try:
    from streamlit_geolocation import streamlit_geolocation
    HAS_GEOLOCATION = True
except ImportError:
    HAS_GEOLOCATION = False

from baseline import nearest_neighbor_baseline
from data_generator import DemoConfig, generate_demo_data, load_points_from_dataframe, DEPOT_LOCATION
from dynamic_routing import DynamicRoutingEngine, GPSTrackerConfig, interpolate_along_route
from forecasting import (
    estimate_vehicles_needed,
    evaluate_all_types,
    generate_waste_history,
    get_demand_forecast,
)
from optimizer import OptimizeConfig, solve_cvrp, solve_cvrp_with_auto_scaling
from routing import DEFAULT_OSRM_BASE_URL, OSRMError, get_osrm_matrices, get_osrm_route_geometry

st.set_page_config(page_title="Tối ưu tuyến thu gom rác - Lê Văn Việt", layout="wide")

# ============================================================================
# HEADER - Ghi rõ phạm vi & giả định nghiên cứu (bắt buộc theo yêu cầu đề tài)
# ============================================================================
st.title("Prototype tối ưu hoá tuyến thu gom chất thải rắn sinh hoạt")

st.markdown(
    """
| Hạng mục | Nội dung |
|---|---|
| **Khu vực thử nghiệm** | Tuyến Lê Văn Việt – TP. Thủ Đức, TP.HCM (khu vực Quận 9 cũ) |
| **Routing engine** | OpenStreetMap + OSRM |
| **Optimization** | Google OR-Tools + Guided Local Search |
| **Real-time traffic** | Chưa xét |
| **Baseline** | Simulated baseline – Nearest Neighbor / Greedy |
"""
)
st.caption(
    "Đây là mô hình nghiên cứu/prototype phục vụ mục đích học thuật (Green Logistics), "
    "KHÔNG phải hệ thống điều hành xe thu gom rác thực tế."
)
st.divider()

# ============================================================================
# SIDEBAR - CẤU HÌNH
# ============================================================================
with st.sidebar:
    st.header("0. Dự báo nhu cầu (Machine Learning)")
    use_ml_forecast = st.checkbox(
        "Dùng demand DỰ BÁO (ML) thay vì demand cố định", value=False,
        help="Dự báo khối lượng rác ngày tới bằng XGBoost/Prophet dựa trên "
             "dữ liệu lịch sử mô phỏng, rồi dùng kết quả này làm demand đầu "
             "vào cho OR-Tools thay vì cột waste_kg cố định.",
    )
    if use_ml_forecast:
        forecast_method = st.selectbox("Phương pháp dự báo", ["xgboost", "prophet"])
        history_days = st.slider("Số ngày lịch sử dùng để huấn luyện", 60, 365, 180)
        forecast_offset = st.slider("Dự báo cho bao nhiêu ngày tới", 1, 14, 1)
        st.caption(
            "Dữ liệu lịch sử được sinh mô phỏng (chưa có dữ liệu cân rác thật). "
            "Việc thay thế bằng dữ liệu lịch sử thực tế thực hiện trong "
            "generate_waste_history() của forecasting.py."
        )

    st.header("1. Dữ liệu")
    data_mode = st.radio("Nguồn dữ liệu", ["Dữ liệu demo (Lê Văn Việt)", "Upload CSV/XLSX"])

    if data_mode == "Dữ liệu demo (Lê Văn Việt)":
        num_points = st.slider("Số điểm thu gom (demo, quanh Lê Văn Việt)", 20, 30, 25)
        seed = st.number_input("Random seed", value=42, step=1)
        use_tw_demo = st.checkbox("Sinh time window demo (VRPTW)", value=False)
        st.caption("Case study quy mô nhỏ: 20–30 điểm thu gom + 1 depot, phù hợp 2–3 xe.")
    else:
        uploaded_file = st.file_uploader("Upload file (CSV hoặc XLSX)", type=["csv", "xlsx"])
        st.caption(
            "Cột bắt buộc: node_id, latitude, longitude, waste_kg. "
            "Tuỳ chọn: service_time, time_window_start, time_window_end, is_depot."
        )

    st.header("2. Routing (OSRM)")
    osrm_base_url = st.text_input("OSRM base URL", value=DEFAULT_OSRM_BASE_URL)
    allow_fallback = st.checkbox(
        "Cho phép fallback Haversine nếu OSRM lỗi (KHÔNG khuyến nghị)", value=False
    )
    if allow_fallback:
        st.warning("Fallback mode – không sử dụng mạng lưới đường thực tế nếu được kích hoạt.")
        fallback_speed = st.slider("Tốc độ giả định cho fallback (km/h)", 10, 50, 25)
    else:
        fallback_speed = 25

    st.header("3. Xe & ràng buộc")
    num_vehicles = st.slider("Số xe tối đa (upper bound cho OR-Tools)", 1, 10, 3)
    st.caption("Case study quy mô nhỏ: mặc định 2-3 xe thu gom.")
    vehicle_capacity_kg = st.number_input("Vehicle capacity (kg)", value=1000, step=50)
    max_route_hours = st.slider("Max route duration (giờ)", 1.0, 8.0, 4.0, step=0.5)

    st.header("4. Thuật toán tối ưu (OR-Tools)")
    use_gls = st.checkbox("Bật Guided Local Search (GLS)", value=True)
    first_solution_strategy = st.selectbox(
        "Chiến lược khởi tạo (initial solution)",
        ["PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"],
    )
    time_limit_sec = st.slider("Thời gian chạy tối ưu (giây)", 5, 120, 20)

    st.header("5. Hệ số tiêu hao & phát thải (có thể chỉnh)")
    fuel_rate_l_per_km = st.number_input("Fuel rate (lít/km)", value=0.35, step=0.01, format="%.2f")
    emission_factor_kg_per_l = st.number_input(
        "Emission factor (kg CO2 / lít nhiên liệu)", value=2.68, step=0.01, format="%.2f"
    )

    run_btn = st.button("Chạy tối ưu", type="primary", use_container_width=True)


# ============================================================================
# LOAD DỮ LIỆU
# ============================================================================
def _load_data() -> pd.DataFrame | None:
    if data_mode == "Dữ liệu demo (Lê Văn Việt)":
        cfg = DemoConfig(num_points=num_points, seed=int(seed), use_time_windows=use_tw_demo)
        return generate_demo_data(cfg)

    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return load_points_from_dataframe(raw)


if "df_points" not in st.session_state:
    st.session_state.df_points = None
if "results" not in st.session_state:
    st.session_state.results = None

df_points = _load_data()
if df_points is not None:
    st.session_state.df_points = df_points

if st.session_state.df_points is None:
    st.info("Vui lòng upload dữ liệu hoặc dùng dữ liệu demo, sau đó nhấn **Chạy tối ưu**.")
    st.stop()

df_points = st.session_state.df_points

# ============================================================================
# DỰ BÁO NHU CẦU (ML) - chạy TRƯỚC khi tối ưu tuyến, nếu được bật ở sidebar
# ============================================================================
if use_ml_forecast:
    st.subheader("📈 Dự báo nhu cầu rác thải (Machine Learning)")
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_forecast_btn = st.button("Chạy dự báo demand", use_container_width=True)
    with col_info:
        st.caption(
            f"Phương pháp: **{forecast_method}** · Lịch sử: {history_days} ngày · "
            f"Dự báo: {forecast_offset} ngày tới."
        )

    if run_forecast_btn:
        with st.spinner("Đang sinh dữ liệu lịch sử và huấn luyện mô hình dự báo..."):
            history_df = generate_waste_history(df_points, days=history_days, seed=int(seed) if data_mode == "Dữ liệu demo (Lê Văn Việt)" else 42)
            demand_forecast = get_demand_forecast(
                df_points, history_days=history_days,
                forecast_day_offset=forecast_offset, method=forecast_method,
            )
            # Chỉ đánh giá MAE/feature importance bằng XGBoost (Prophet không
            # xuất feature_importance vì không dùng feature engineering dạng bảng).
            cutoff_date = (history_df["date"].max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            eval_df = evaluate_all_types(history_df, cutoff_date)

        st.session_state.demand_forecast = demand_forecast
        st.session_state.forecast_eval = eval_df

    if "demand_forecast" in st.session_state:
        demand_forecast = st.session_state.demand_forecast
        total_kg, suggested_vehicles = estimate_vehicles_needed(
            demand_forecast, vehicle_capacity_kg=vehicle_capacity_kg
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Tổng demand dự báo (kg)", f"{total_kg:,.0f}")
        m2.metric("Số xe đề xuất (dựa trên dự báo)", suggested_vehicles)
        m3.metric("Số xe đang cấu hình (sidebar)", num_vehicles)
        if suggested_vehicles > num_vehicles:
            st.warning(
                f"⚠️ Dự báo cho thấy cần tối thiểu **{suggested_vehicles} xe** để "
                f"đủ tải, nhưng sidebar đang chỉ cấu hình {num_vehicles} xe. "
                "Hãy tăng 'Số xe tối đa' ở mục 3 trước khi chạy tối ưu."
            )
        else:
            st.success(
                f"✅ Số xe hiện cấu hình ({num_vehicles}) đủ đáp ứng demand dự báo "
                f"(cần tối thiểu {suggested_vehicles} xe)."
            )

        with st.expander("Chi tiết demand dự báo theo từng điểm + loại rác", expanded=False):
            forecast_table = pd.DataFrame(demand_forecast).T
            forecast_table.index.name = "node_id"
            st.dataframe(forecast_table, use_container_width=True)

        with st.expander("Độ chính xác mô hình (MAE) + Feature quan trọng nhất", expanded=False):
            st.dataframe(st.session_state.forecast_eval, use_container_width=True, hide_index=True)
            st.caption(
                "MAE (Mean Absolute Error) càng thấp càng chính xác. Feature quan "
                "trọng nhất cho biết yếu tố nào ảnh hưởng nhiều nhất đến khối lượng "
                "rác dự báo được (VD: is_holiday ảnh hưởng mạnh đến rác hữu cơ dịp Tết)."
            )

        apply_forecast = st.checkbox(
            "✅ Dùng demand dự báo này làm đầu vào cho VRP (thay cho waste_kg cố định)",
            value=False,
        )
        if apply_forecast:
            df_points = df_points.copy()
            df_points["waste_kg"] = df_points["node_id"].map(
                lambda nid: demand_forecast.get(nid, {}).get("total_kg", 0.0)
            )
            df_points.loc[df_points["is_depot"], "waste_kg"] = 0.0
            st.session_state.df_points = df_points
            st.info("Đã cập nhật waste_kg theo demand dự báo. Nhấn **Chạy tối ưu** để áp dụng.")

    st.divider()

with st.expander("Xem dữ liệu điểm thu gom", expanded=False):
    st.dataframe(df_points, use_container_width=True)

# ============================================================================
# CHẠY PIPELINE KHI NHẤN NÚT
# ============================================================================
if run_btn:
    coords = tuple(zip(df_points["latitude"], df_points["longitude"]))
    demands = df_points["waste_kg"].tolist()
    service_times_s = (df_points["service_time"] * 60).tolist()
    time_windows_s = list(
        zip(df_points["time_window_start"] * 60, df_points["time_window_end"] * 60)
    )
    use_tw = df_points["time_window_start"].sum() > 0 or (
        data_mode == "Dữ liệu demo (Lê Văn Việt)" and use_tw_demo
    )

    with st.spinner("Đang gọi OSRM để lấy ma trận khoảng cách/thời gian đường bộ..."):
        try:
            matrix_result = get_osrm_matrices(
                coords,
                base_url=osrm_base_url,
                allow_haversine_fallback=allow_fallback,
                fallback_avg_speed_kmh=fallback_speed,
            )
        except OSRMError as exc:
            st.error(str(exc))
            st.stop()

    if matrix_result.source == "HAVERSINE_FALLBACK":
        st.warning(matrix_result.warning)
    else:
        st.success("Đã lấy ma trận khoảng cách/thời gian từ OSRM (Routing: OpenStreetMap + OSRM).")

    dist_m = matrix_result.distance_matrix_m
    dur_s = matrix_result.duration_matrix_s

    depot_index = int(df_points.index[df_points["is_depot"]][0])

    # ---------------- BASELINE ----------------
    with st.spinner("Đang xây dựng baseline (Nearest Neighbor / Greedy)..."):
        try:
            baseline_routes = nearest_neighbor_baseline(
                dist_m, dur_s, demands, service_times_s,
                vehicle_capacity_kg=vehicle_capacity_kg,
                depot_index=depot_index,
                max_route_time_s=max_route_hours * 3600,
                max_vehicles=max(num_vehicles, 20),
            )
        except RuntimeError as exc:
            st.error(f"Baseline thất bại: {exc}")
            st.stop()

    # ---------------- OPTIMIZED (OR-TOOLS + GLS) ----------------
    with st.spinner("Đang chạy OR-Tools + Guided Local Search..."):
        opt_config = OptimizeConfig(
            num_vehicles=max(num_vehicles, len(baseline_routes)),
            vehicle_capacity_kg=vehicle_capacity_kg,
            depot_index=depot_index,
            use_gls=use_gls,
            first_solution_strategy=first_solution_strategy,
            time_limit_sec=time_limit_sec,
            max_route_time_s=max_route_hours * 3600,
            use_time_windows=use_tw,
            time_windows_s=time_windows_s,
        )
        optimized_routes, solved, msg = solve_cvrp(dist_m, dur_s, demands, service_times_s, opt_config)

    if not solved:
        st.error(msg)
        st.stop()

    st.session_state.results = {
        "df_points": df_points,
        "coords": coords,
        "matrix_result": matrix_result,
        "baseline_routes": baseline_routes,
        "optimized_routes": optimized_routes,
        "depot_index": depot_index,
        "osrm_base_url": osrm_base_url,
        "fuel_rate_l_per_km": fuel_rate_l_per_km,
        "emission_factor_kg_per_l": emission_factor_kg_per_l,
        "demands": demands,
        "service_times_s": service_times_s,
    }

# ============================================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================================
if st.session_state.results is None:
    st.stop()

res = st.session_state.results
df_points = res["df_points"]
coords = res["coords"]
baseline_routes = res["baseline_routes"]
optimized_routes = res["optimized_routes"]
depot_index = res["depot_index"]
fuel_rate = res["fuel_rate_l_per_km"]
emission_factor = res["emission_factor_kg_per_l"]


def _aggregate(routes) -> dict:
    total_distance_km = sum(r.total_distance_m for r in routes) / 1000.0
    travel_time_min = sum(r.travel_time_s for r in routes) / 60.0
    service_time_min = sum(r.service_time_s for r in routes) / 60.0
    total_time_min = sum(r.total_route_time_s for r in routes) / 60.0
    num_vehicles_used = len(routes)
    total_waste_kg = sum(r.collected_waste_kg for r in routes)
    avg_util = (
        sum(r.capacity_utilization_pct for r in routes) / len(routes) if routes else 0.0
    )
    fuel_l = total_distance_km * fuel_rate
    co2_kg = fuel_l * emission_factor
    return {
        "Total distance (km)": round(total_distance_km, 2),
        "Travel time (min)": round(travel_time_min, 1),
        "Service time (min)": round(service_time_min, 1),
        "Total route time (min)": round(total_time_min, 1),
        "Number of vehicles": num_vehicles_used,
        "Total waste collected (kg)": round(total_waste_kg, 1),
        "Average capacity utilization (%)": round(avg_util, 1),
        "Estimated fuel consumption (L)": round(fuel_l, 2),
        "Estimated CO2 emissions (kg)": round(co2_kg, 2),
    }


baseline_kpi = _aggregate(baseline_routes)
optimized_kpi = _aggregate(optimized_routes)


def _reduction(base, opt):
    if base == 0:
        return 0.0
    return round((base - opt) / base * 100, 1)


kpi_rows = []
for key in baseline_kpi:
    row = {"Chỉ tiêu": key, "Baseline": baseline_kpi[key], "Optimized": optimized_kpi[key]}
    if key in (
        "Total distance (km)", "Total route time (min)",
        "Estimated CO2 emissions (kg)", "Estimated fuel consumption (L)",
    ):
        row["Reduction (%)"] = _reduction(baseline_kpi[key], optimized_kpi[key])
    else:
        row["Reduction (%)"] = "-"
    kpi_rows.append(row)

st.subheader("So sánh KPI: Baseline vs Optimized")
st.caption(
    "Baseline = tuyến cơ sở mô phỏng bằng heuristic Nearest Neighbor/Greedy "
    f"(nguồn khoảng cách/thời gian: {res['matrix_result'].source}). "
    "CO2 emissions là giá trị ước tính (Estimated CO2 emissions), không phải đo trực tiếp."
)
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

# ---------------- Chi tiết từng tuyến ----------------
node_ids = df_points["node_id"].tolist()


def _route_label(route) -> str:
    names = [node_ids[i] for i in route.node_sequence]
    return " → ".join(names)


col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline routes**")
    for r in baseline_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )
with col_b:
    st.markdown("**Optimized routes (OR-Tools + GLS)**")
    for r in optimized_routes:
        st.text(f"Vehicle {r.vehicle_id}: {_route_label(r)}")
        st.caption(
            f"Distance: {r.total_distance_m/1000:.2f} km | "
            f"Total time: {r.total_route_time_s/60:.1f} phút | "
            f"Waste: {r.collected_waste_kg:.1f} kg | "
            f"Utilization: {r.capacity_utilization_pct:.1f}%"
        )

st.divider()

# ============================================================================
# BẢN ĐỒ (Folium + OSRM road geometry thực tế)
# ============================================================================
st.subheader("Bản đồ tuyến (road geometry thực tế từ OSRM)")

center_lat, center_lon = DEPOT_LOCATION
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# Depot & các điểm thu gom
for idx, row in df_points.iterrows():
    if row["is_depot"]:
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup="DEPOT",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(fmap)
    else:
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=5,
            popup=f"{row['node_id']} - {row['waste_kg']:.0f} kg",
            color="#555555",
            fill=True,
            fill_opacity=0.8,
        ).add_to(fmap)


def _draw_routes(routes, color, label_prefix):
    for r in routes:
        ordered_coords = tuple(coords[i] for i in r.node_sequence)
        geometry = get_osrm_route_geometry(ordered_coords, base_url=res["osrm_base_url"])
        if not geometry:
            # OSRM route service không khả dụng cho tuyến này -> vẽ tạm bằng
            # đường nối các điểm (KHÔNG phải road geometry thực tế), có ghi chú.
            geometry = list(ordered_coords)
            dash = "5, 10"
        else:
            dash = None
        folium.PolyLine(
            geometry,
            color=color,
            weight=4,
            opacity=0.8,
            dash_array=dash,
            tooltip=f"{label_prefix} - Vehicle {r.vehicle_id} ({r.total_distance_m/1000:.2f} km)",
        ).add_to(fmap)


_draw_routes(baseline_routes, "#1f77b4", "Baseline")
_draw_routes(optimized_routes, "#d62728", "Optimized")

st.caption(
    "🔵 Xanh dương = Baseline (Nearest Neighbor/Greedy) · 🔴 Đỏ = Optimized (OR-Tools + GLS). "
    "Nét đứt (nếu có) nghĩa là OSRM Route Service không trả về được geometry cho đoạn đó."
)
st_folium(fmap, use_container_width=True, height=600, returned_objects=[])

st.divider()

# ============================================================================
# DYNAMIC ROUTING (GPS TỰ ĐỘNG) - KHÔNG cần tài xế bấm nút xác nhận
# ============================================================================
st.subheader("🛰️ Dynamic Routing – GPS tự động (Auto completion & Auto re-optimize)")
st.caption(
    "Xe được theo dõi qua GPS điện thoại. Khi xe vào bán kính điểm thu gom và đứng đủ lâu "
    "→ điểm tự động chuyển pending → completed. Khi xe vào bán kính DEPOT và đứng đủ lâu "
    "→ hệ thống tự xác nhận 'xe đã về DEPOT' và TỰ ĐỘNG tái tối ưu (OR-Tools + GLS) cho các "
    "điểm còn pending, KHÔNG cần tài xế bấm bất kỳ nút xác nhận nào."
)

node_ids_full = df_points["node_id"].tolist()
index_of_node = {nid: i for i, nid in enumerate(node_ids_full)}
depot_node_id = df_points.loc[df_points["is_depot"], "node_id"].iloc[0]
dist_m_full = res["matrix_result"].distance_matrix_m
dur_s_full = res["matrix_result"].duration_matrix_s
demands_full = res["demands"]
service_times_s_full = res["service_times_s"]

vehicle_options = {f"Vehicle {r.vehicle_id} ({r.num_stops} điểm)": r for r in optimized_routes}

with st.expander("⚙️ Cấu hình Dynamic Routing", expanded=st.session_state.get("gps_enabled", False)):
    gps_enabled = st.checkbox("Bật theo dõi GPS tự động cho 1 xe", key="gps_enabled")
    chosen_label = st.selectbox("Chọn xe để theo dõi GPS", list(vehicle_options.keys()))
    gps_mode = st.radio(
        "Nguồn GPS",
        ["Mô phỏng GPS (demo/test, không cần thiết bị)", "GPS thực từ điện thoại (thử nghiệm)"],
    )
    col1, col2 = st.columns(2)
    with col1:
        geofence_radius_m = st.slider("Bán kính auto-completion & depot (m)", 30, 50, 40)
    with col2:
        dwell_seconds = st.slider("Thời gian lưu tối thiểu trong vùng (giây)", 5, 30, 10)

    if gps_mode.startswith("Mô phỏng"):
        sim_speed_kmh = st.slider("Tốc độ xe mô phỏng (km/h)", 5, 40, 20)
        sim_accel = st.slider("Tăng tốc mô phỏng (số giây mô phỏng / lần refresh)", 1, 20, 6)
    else:
        sim_speed_kmh, sim_accel = 20, 6
        if not HAS_GEOLOCATION:
            st.error(
                "Chưa cài được package streamlit-geolocation trong môi trường này. "
                "Hãy `pip install streamlit-geolocation` rồi chạy lại."
            )
        st.info(
            "Lưu ý: trình duyệt yêu cầu quyền định vị và (tuỳ thiết bị/trình duyệt) có thể cần "
            "tap lại nút định vị để cấp phép — đây là giới hạn bảo mật của trình duyệt, "
            "không phải giới hạn của logic auto-completion/auto re-optimize."
        )

    if st.button("🔄 Khởi tạo / Reset theo dõi GPS cho xe đã chọn"):
        chosen_route = vehicle_options[chosen_label]
        points_for_engine = [
            {
                "node_id": node_ids_full[i],
                "latitude": df_points.iloc[i]["latitude"],
                "longitude": df_points.iloc[i]["longitude"],
            }
            for i in chosen_route.node_sequence
            if not df_points.iloc[i]["is_depot"]
        ]
        engine = DynamicRoutingEngine(
            points_for_engine,
            depot_latlon=DEPOT_LOCATION,
            config=GPSTrackerConfig(
                completion_radius_m=geofence_radius_m,
                depot_radius_m=geofence_radius_m,
                dwell_seconds_required=dwell_seconds,
            ),
        )
        active_coords = tuple(coords[i] for i in chosen_route.node_sequence)
        active_geometry = get_osrm_route_geometry(active_coords, base_url=res["osrm_base_url"])
        if not active_geometry:
            active_geometry = list(active_coords)

        st.session_state.gps_engine = engine
        st.session_state.gps_active_route_nodes = list(chosen_route.node_sequence)
        st.session_state.gps_active_geometry = active_geometry
        st.session_state.gps_sim_time_s = 0.0
        st.session_state.gps_sim_progress_m = 0.0
        st.session_state.gps_event_log = ["Đã khởi tạo theo dõi GPS cho " + chosen_label]
        st.session_state.gps_tour_finished = False
        st.rerun()

if gps_enabled and "gps_engine" in st.session_state:
    engine: DynamicRoutingEngine = st.session_state.gps_engine

    # Tick tự động ~1.5s/lần để mô phỏng/đọc GPS liên tục (không cần bấm nút)
    st_autorefresh(interval=1500, key="gps_autorefresh_tick")

    def _handle_event(ev: dict):
        if ev["newly_completed"]:
            for nid in ev["newly_completed"]:
                st.session_state.gps_event_log.append(f"✅ Auto-completed: {nid}")
        if ev["depot_confirmed"]:
            st.session_state.gps_event_log.append("🏠 Auto depot detected: xe đã về DEPOT")
            pending_ids = engine.pending_node_ids()
            if not pending_ids:
                st.session_state.gps_tour_finished = True
                st.session_state.gps_event_log.append("🎉 Đã hoàn thành toàn bộ tuyến.")
                return
            # ---- AUTO RE-OPTIMIZATION: chỉ các điểm pending, DEPOT là điểm xuất phát ----
            sub_indices = [index_of_node[depot_node_id]] + [index_of_node[nid] for nid in pending_ids]
            sub_dist = [[dist_m_full[a][b] for b in sub_indices] for a in sub_indices]
            sub_dur = [[dur_s_full[a][b] for b in sub_indices] for a in sub_indices]
            sub_demands = [demands_full[i] for i in sub_indices]
            sub_service = [service_times_s_full[i] for i in sub_indices]

            reopt_cfg = OptimizeConfig(
                num_vehicles=1,  # cùng 1 xe vật lý tiếp tục hành trình
                vehicle_capacity_kg=vehicle_capacity_kg,
                depot_index=0,
                use_gls=use_gls,
                first_solution_strategy=first_solution_strategy,
                time_limit_sec=min(time_limit_sec, 15),
                max_route_time_s=max_route_hours * 3600,
            )
            new_routes, solved, msg, _n = solve_cvrp_with_auto_scaling(
                sub_dist, sub_dur, sub_demands, sub_service, reopt_cfg, max_extra_vehicles=0
            )
            if solved and new_routes:
                new_route_global_nodes = [sub_indices[i] for i in new_routes[0].node_sequence]
                st.session_state.gps_active_route_nodes = new_route_global_nodes
                new_active_coords = tuple(coords[i] for i in new_route_global_nodes)
                new_geom = get_osrm_route_geometry(new_active_coords, base_url=res["osrm_base_url"])
                st.session_state.gps_active_geometry = new_geom or list(new_active_coords)
                st.session_state.gps_sim_progress_m = 0.0
                engine.sync_pending_after_reoptimize(pending_ids)
                st.session_state.gps_event_log.append(
                    f"🔁 Auto re-optimized: DEPOT → {len(pending_ids)} điểm pending còn lại "
                    f"({new_routes[0].total_distance_m/1000:.2f} km)"
                )
            else:
                st.session_state.gps_event_log.append(f"⚠️ Re-optimize thất bại: {msg}")

    if not st.session_state.get("gps_tour_finished", False):
        if gps_mode.startswith("Mô phỏng"):
            speed_mps = sim_speed_kmh * 1000 / 3600
            geometry = st.session_state.gps_active_geometry
            for _ in range(int(sim_accel)):
                st.session_state.gps_sim_time_s += 1.0
                st.session_state.gps_sim_progress_m += speed_mps * 1.0
                lat, lon, finished = interpolate_along_route(geometry, st.session_state.gps_sim_progress_m)
                if lat is None:
                    break
                ev = engine.update_position(lat, lon, ts=st.session_state.gps_sim_time_s)
                _handle_event(ev)
                if st.session_state.get("gps_tour_finished", False):
                    break
        else:
            if HAS_GEOLOCATION:
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                    ev = engine.update_position(loc["latitude"], loc["longitude"], ts=time.time())
                    _handle_event(ev)

    # ---- Hiển thị trạng thái ----
    col_map, col_status = st.columns([2, 1])
    with col_status:
        st.markdown("**Trạng thái điểm thu gom**")
        status_rows = [
            {"node_id": nid, "status": s.status}
            for nid, s in engine.point_status.items()
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True, height=250)
        st.metric("Điểm còn pending", len(engine.pending_node_ids()))
        if st.session_state.get("gps_tour_finished"):
            st.success("Xe đã hoàn thành toàn bộ tuyến (tất cả điểm completed).")
        with st.expander("Nhật ký sự kiện", expanded=True):
            for line in st.session_state.gps_event_log[-15:][::-1]:
                st.text(line)

    with col_map:
        gmap = folium.Map(location=[DEPOT_LOCATION[0], DEPOT_LOCATION[1]], zoom_start=15, tiles="cartodbpositron")
        folium.Marker(DEPOT_LOCATION, popup="DEPOT", icon=folium.Icon(color="black", icon="home")).add_to(gmap)
        for nid, s in engine.point_status.items():
            plat, plon = engine.point_coords[nid]
            color = "#2ca02c" if s.status == "completed" else "#ff7f0e"
            folium.CircleMarker(
                [plat, plon], radius=6, color=color, fill=True, fill_opacity=0.9,
                popup=f"{nid} - {s.status}",
            ).add_to(gmap)
        folium.PolyLine(
            st.session_state.gps_active_geometry, color="#9467bd", weight=4, opacity=0.7,
            tooltip="Tuyến đang chạy (auto re-optimize khi về Depot)",
        ).add_to(gmap)
        if engine.current_position:
            folium.Marker(
                engine.current_position,
                popup="Xe (GPS hiện tại)",
                icon=folium.Icon(color="blue", icon="truck", prefix="fa"),
            ).add_to(gmap)
        st_folium(gmap, use_container_width=True, height=500, returned_objects=[], key="gps_map")
elif gps_enabled:
    st.info("Nhấn **'Khởi tạo / Reset theo dõi GPS cho xe đã chọn'** ở trên để bắt đầu.")
