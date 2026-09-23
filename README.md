# Prototype tối ưu tuyến thu gom chất thải rắn sinh hoạt
## Khu vực nghiên cứu: Tuyến Lê Văn Việt và lân cận, TP. Thủ Đức, TP.HCM

## 1. Kiến trúc code

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, điều phối toàn bộ luồng xử lý |
| `routing.py` | Toàn bộ chức năng OSM/OSRM (distance/duration matrix, road geometry), cache, xử lý lỗi |
| `baseline.py` | Baseline heuristic Nearest Neighbor / Greedy (có capacity, service time, depot) |
| `optimizer.py` | OR-Tools CVRP/VRPTW + Guided Local Search |
| `data_generator.py` | Sinh dữ liệu demo trong phạm vi Lê Văn Việt; chuẩn hoá dữ liệu upload |
| `requirements.txt` | Thư viện cần cài |

## 2. Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại địa chỉ Streamlit hiển thị (mặc định `http://localhost:8501`).

> **Lưu ý về OSRM:** Ứng dụng mặc định gọi OSRM demo server công khai
> (`https://router.project-osrm.org`). Server này có giới hạn tốc độ và chỉ
> phù hợp cho mục đích thử nghiệm/nghiên cứu. Nếu cần dùng ổn định/số lượng
> lớn, nên tự dựng OSRM server (Docker + dữ liệu OSM khu vực TP.HCM) và đổi
> "OSRM base URL" trong sidebar sang địa chỉ server riêng.

## 4. Chuẩn bị file CSV/XLSX (nếu không dùng dữ liệu demo)

Cột bắt buộc:

| Cột | Ý nghĩa |
|---|---|
| `node_id` | Mã điểm (duy nhất) |
| `latitude` | Vĩ độ |
| `longitude` | Kinh độ |
| `waste_kg` | Khối lượng rác phát sinh tại điểm (kg) |

Cột tuỳ chọn:

| Cột | Ý nghĩa | Mặc định nếu thiếu |
|---|---|---|
| `service_time` | Thời gian phục vụ tại điểm (phút) | 5 |
| `time_window_start` | Bắt đầu khung giờ được phép thu gom (phút, tính từ 00:00) | 0 |
| `time_window_end` | Kết thúc khung giờ (phút) | 1440 |
| `is_depot` | `True` cho đúng 1 hàng là depot | hàng đầu tiên |

Tất cả toạ độ nên nằm trong/quanh khu vực Lê Văn Việt, TP. Thủ Đức để đúng
phạm vi nghiên cứu đã xác định.

## 5. Luồng xử lý (pipeline)

```
Data (demo hoặc upload)
   -> OSM/OSRM (Table Service)
   -> Distance matrix (m) + Time matrix (s)   [ma trận CHÍNH cho toàn bộ mô hình]
   -> Baseline (Nearest Neighbor / Greedy, cùng ma trận OSRM)
   -> OR-Tools (CVRP / VRPTW)
   -> Guided Local Search (metaheuristic)
   -> Optimized Routes
   -> OSRM Route Service -> Road geometry thực tế cho từng tuyến
   -> Folium Map (baseline vs optimized)
   -> Bảng KPI (distance, time, số xe, capacity utilization, fuel, CO2)
   -> So sánh % giảm (distance / time / CO2)
```

## 6. Các nguyên tắc đã áp dụng theo yêu cầu

- **Không dùng Haversine làm khoảng cách chính** – chỉ dùng khi người dùng
  chủ động bật "cho phép fallback Haversine", và giao diện sẽ hiển thị rõ
  cảnh báo "Fallback mode – không sử dụng mạng lưới đường thực tế".
- **Không dùng `distance / average_speed` làm travel time chính** – thời
  gian di chuyển luôn lấy từ ma trận `duration` của OSRM Table Service.
- **Baseline và Optimized dùng chung một ma trận OSRM** để đảm bảo so sánh
  công bằng.
- **CO2 là giá trị ước tính** ("Estimated CO2 emissions"), tính từ
  `distance × fuel_rate × emission_factor`; các hệ số này có thể chỉnh trong
  sidebar.
- **Không âm thầm chuyển sang Haversine khi OSRM lỗi** – mặc định báo lỗi rõ
  ràng: *"Không thể lấy dữ liệu mạng lưới đường từ OSRM. Vui lòng thử lại."*
- **Cache OSRM** bằng `st.cache_data` cho distance/duration matrix và cho
  road geometry, tránh gọi lại OSRM không cần thiết mỗi lần Streamlit rerun.
- **Dữ liệu demo** chỉ được sinh quanh các waypoint dọc tuyến Lê Văn Việt
  (20–30 điểm + 1 depot, mặc định 2–3 xe), bán kính lệch tối đa ~200m để mô
  phỏng các hẻm/đường nhánh kết nối trực tiếp với trục chính — không rải ra
  toàn TP.HCM/TP. Thủ Đức.
- Nhãn khu vực hiển thị trên giao diện: **"Khu vực thử nghiệm: Tuyến Lê Văn
  Việt – TP. Thủ Đức, TP.HCM"**.

## 7. Giới hạn của prototype (đúng với mục tiêu đề tài sinh viên)

- Chưa xét traffic thời gian thực.
- Toạ độ waypoint dọc Lê Văn Việt trong `data_generator.py` là toạ độ tham
  khảo/xấp xỉ để mô phỏng khu vực nghiên cứu, không phải kết quả khảo sát
  thực địa chính xác từng mét — nếu dùng cho báo cáo chính thức, nên thay
  bằng toạ độ khảo sát thực tế (qua file CSV/XLSX upload).
- OR-Tools + Guided Local Search là thuật toán tối ưu duy nhất được sử dụng,
  không thay thế bằng thuật toán AI khác.

## 8. Dynamic Routing (GPS tự động) — tính năng mới

Ở cuối trang có section **"🛰️ Dynamic Routing – GPS tự động"** theo dõi 1 xe
theo thời gian thực và tự động re-optimize, KHÔNG cần tài xế bấm nút xác nhận:

- **Auto completion**: xe vào bán kính 30–50m quanh 1 điểm thu gom và đứng
  yên trong vùng đó tối thiểu 10 giây (mặc định, có thể chỉnh) → điểm tự
  chuyển `pending → completed`.
- **Auto depot detection**: xe vào bán kính 30–50m quanh DEPOT và đứng đủ
  thời gian → hệ thống tự xác nhận "xe đã về DEPOT", không cần bấm nút.
- **Auto re-optimization**: ngay khi xác nhận về DEPOT, hệ thống lấy toàn bộ
  điểm còn `pending`, đặt DEPOT làm điểm xuất phát, tự động chạy lại
  OR-Tools + Guided Local Search và hiển thị tuyến mới trên bản đồ.
- **Không suy luận "xe đầy" từ GPS**: tín hiệu duy nhất để tái tối ưu là
  "xe đã về DEPOT" (dwell tại depot đủ lâu). Không có cảm biến tải trọng
  trong phạm vi prototype này.

Có 2 nguồn GPS:

1. **Mô phỏng GPS (demo/test)** — xe di chuyển ảo dọc theo road geometry của
   tuyến hiện tại với tốc độ và hệ số tăng tốc do người dùng chọn. Đây là
   cách khuyến nghị để demo/kiểm chứng logic auto-completion/auto
   re-optimize mà không cần xe thật.
2. **GPS thực từ điện thoại (thử nghiệm)** — dùng package
   `streamlit-geolocation` để đọc vị trí trình duyệt. **Giới hạn quan trọng**:
   trình duyệt yêu cầu quyền định vị và, tuỳ thiết bị/trình duyệt, có thể cần
   tương tác lại để cấp phép hoặc lấy vị trí mới — đây là giới hạn bảo mật
   của trình duyệt web (không có API JavaScript nào theo dõi GPS nền hoàn
   toàn im lặng trên mọi trình duyệt/thiết bị di động). Nếu cần độ tin cậy
   cao hơn cho vận hành thực tế, nên dùng app di động gốc (native) gửi GPS
   định kỳ qua API riêng.

Module logic (`dynamic_routing.py`) hoàn toàn tách biệt khỏi Streamlit nên có
thể unit-test độc lập (xem cách test trong quá trình phát triển) và tái sử
dụng cho backend khác nếu cần mở rộng sau này.
