"""
simulation.py
--------------
FLEET REAL-TIME SIMULATION ENGINE — mô phỏng vận hành nhiều xe thu gom cùng
lúc theo thời gian, dùng để bổ sung cho các module tĩnh đã có (forecasting,
clustering, optimizer, routing) theo đúng phần "14. REAL-TIME SIMULATION" →
"20. RE-OPTIMIZATION ENGINE" của đặc tả gốc.

Thiết kế: module này KHÔNG import Streamlit / OR-Tools / requests, để có thể
unit-test độc lập (giống dynamic_routing.py). Việc gọi OSRM (lấy road
geometry) và OR-Tools (giải lại CVRP khi cần re-optimize) do lớp gọi (app.py)
thực hiện rồi "bơm" kết quả ngược vào FleetSimulator qua assign_route() /
redirect_to_depot().

TRẠNG THÁI XE (bắt buộc theo đặc tả mục 14):
    IDLE, MOVING, COLLECTING, FULL, RETURNING_TO_DEPOT, BROKEN, BLOCKED, COMPLETED

Quy tắc cốt lõi:
- Xe di chuyển dọc theo "leg" (đoạn road-geometry OSRM thực tế giữa 2 điểm
  liên tiếp trong tuyến), KHÔNG nối thẳng toạ độ (mục 15).
- Nếu điểm tiếp theo sẽ làm vượt capacity -> bỏ điểm đó lại (pending), xe
  quay depot, xả tải, sau đó cần re-optimize phần còn lại (mục 16).
- Nếu actual_waste tại điểm > predicted_waste và làm vượt capacity -> xử lý
  y hệt quy tắc mục 16 (mục 17).
- Truck hỏng (BROKEN): loại khỏi fleet hoạt động, các điểm xe đó chưa phục vụ
  chuyển vào pool chờ tái phân bổ (mục 18).
- Road blockage: đánh dấu 1 cạnh (node_a, node_b) bị chặn; xe đang dùng đúng
  leg đó chuyển BLOCKED, cần re-route (mục 19).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dynamic_routing import haversine_m, interpolate_along_route

# ---------------------------------------------------------------------------
STATE_IDLE = "IDLE"
STATE_MOVING = "MOVING"
STATE_COLLECTING = "COLLECTING"
STATE_FULL = "FULL"
STATE_RETURNING_TO_DEPOT = "RETURNING_TO_DEPOT"
STATE_BROKEN = "BROKEN"
STATE_BLOCKED = "BLOCKED"
STATE_COMPLETED = "COMPLETED"

ALL_STATES = [
    STATE_IDLE, STATE_MOVING, STATE_COLLECTING, STATE_FULL,
    STATE_RETURNING_TO_DEPOT, STATE_BROKEN, STATE_BLOCKED, STATE_COMPLETED,
]

# Khung giờ hoạt động bắt buộc của xe thu gom (phút, tính từ 00:00).
DEFAULT_OPERATING_WINDOWS_MIN = [(6 * 60, 9 * 60), (16 * 60, 20 * 60)]


def is_within_operating_windows(clock_min: float, windows: list | None = None) -> bool:
    windows = windows or DEFAULT_OPERATING_WINDOWS_MIN
    clock_min = clock_min % (24 * 60)
    return any(start <= clock_min < end for start, end in windows)


def format_hhmm(clock_min: float) -> str:
    clock_min = int(clock_min) % (24 * 60)
    return f"{clock_min // 60:02d}:{clock_min % 60:02d}"


# ---------------------------------------------------------------------------
@dataclass
class RouteLeg:
    """1 đoạn road-geometry thực tế (OSRM) giữa 2 node liên tiếp trong tuyến."""
    from_node: int
    to_node: int
    geometry: list          # [(lat, lon), ...] theo road network thực tế
    length_m: float = 0.0
    duration_s: float = 0.0  # ước lượng thời gian di chuyển (từ ma trận OSRM gốc)

    def edge_key(self) -> tuple:
        a, b = self.from_node, self.to_node
        return (a, b) if a <= b else (b, a)


def _polyline_length_m(points: list) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total


def build_route_legs(node_sequence: list, geometry_fetch, duration_lookup=None) -> list:
    """Xây danh sách RouteLeg cho 1 tuyến [depot, n1, n2, ..., depot].

    geometry_fetch(from_node, to_node) -> list[(lat, lon)] (road geometry
    thực tế, ví dụ gọi OSRM Route Service cho đúng 2 điểm liên tiếp).
    duration_lookup(from_node, to_node) -> giây (tuỳ chọn, mặc định suy ra
    NaN nếu không cung cấp — chỉ dùng để hiển thị, KHÔNG dùng để tính chuyển
    động, vì chuyển động dựa trên tốc độ mô phỏng + chiều dài polyline).
    """
    legs = []
    for i in range(len(node_sequence) - 1):
        a, b = node_sequence[i], node_sequence[i + 1]
        geometry = geometry_fetch(a, b) or []
        length_m = _polyline_length_m(geometry) if len(geometry) >= 2 else 0.0
        duration_s = duration_lookup(a, b) if duration_lookup else 0.0
        legs.append(RouteLeg(from_node=a, to_node=b, geometry=geometry, length_m=length_m, duration_s=duration_s))
    return legs


# ---------------------------------------------------------------------------
@dataclass
class Truck:
    vehicle_id: int
    capacity_kg: float
    speed_kmh: float = 20.0
    status: str = STATE_IDLE

    legs: list = field(default_factory=list)   # RouteLeg hiện đang được gán
    leg_idx: int = 0
    progress_m: float = 0.0                     # tiến độ trong leg hiện tại

    current_load_kg: float = 0.0
    position: tuple | None = None                # (lat, lon) hiện tại

    collect_timer_s: float = 0.0
    pending_nodes: list = field(default_factory=list)   # node đã gán, chưa phục vụ
    served_nodes: list = field(default_factory=list)

    total_distance_m: float = 0.0
    total_active_time_s: float = 0.0
    needs_reopt: bool = False   # True khi vừa về depot và còn pending -> chờ app.py re-route

    @property
    def capacity_utilization_pct(self) -> float:
        if self.capacity_kg <= 0:
            return 0.0
        return 100.0 * self.current_load_kg / self.capacity_kg

    @property
    def current_node(self):
        if not self.legs:
            return None
        if self.leg_idx == 0 and self.status in (STATE_IDLE,):
            return self.legs[0].from_node
        prev_leg = self.legs[min(self.leg_idx, len(self.legs) - 1)]
        return prev_leg.from_node

    @property
    def next_node(self):
        if not self.legs or self.leg_idx >= len(self.legs):
            return None
        return self.legs[self.leg_idx].to_node


# ---------------------------------------------------------------------------
@dataclass
class SimEvent:
    ts_min: float
    text: str


class FleetSimulator:
    """Engine điều phối nhiều xe cùng lúc, theo "sim clock" tính bằng phút,
    bắt đầu từ 06:00 (360 phút) theo mặc định."""

    def __init__(
        self,
        depot_index: int,
        depot_coords: tuple,
        demands_predicted: dict,          # {node_index: predicted_kg}
        service_times_s: dict,            # {node_index: giây phục vụ}
        vehicle_capacity_kg: float,
        start_clock_min: float = 360.0,
        operating_windows_min: list | None = None,
        full_load_threshold_pct: float = 99.5,
    ):
        self.depot_index = depot_index
        self.depot_coords = depot_coords
        self.demands_predicted = dict(demands_predicted)
        self.demands_actual = dict(demands_predicted)  # mặc định = predicted, có thể ghi đè
        self.service_times_s = dict(service_times_s)
        self.vehicle_capacity_kg = vehicle_capacity_kg
        self.operating_windows_min = operating_windows_min or DEFAULT_OPERATING_WINDOWS_MIN
        self.full_load_threshold_pct = full_load_threshold_pct

        self.clock_min = start_clock_min
        self.running = False

        self.trucks: dict[int, Truck] = {}
        self.blocked_edges: set = set()
        self.unassigned_pending: list = []   # node index chưa gán cho xe nào (do broken/reopt)

        self.event_log: list = []
        self._log(f"Simulation khởi tạo lúc {format_hhmm(self.clock_min)}")

    # ------------------------------------------------------------ helpers
    def _log(self, text: str):
        self.event_log.append(SimEvent(ts_min=self.clock_min, text=text))

    def log(self, text: str):
        """Public wrapper so calling code (e.g. app.py) can add its own
        events (re-optimize failures, external triggers) to the same log."""
        self._log(text)

    def recent_events(self, n: int = 20) -> list:
        return self.event_log[-n:][::-1]

    def add_truck(self, vehicle_id: int, capacity_kg: float, speed_kmh: float = 20.0) -> Truck:
        t = Truck(vehicle_id=vehicle_id, capacity_kg=capacity_kg, speed_kmh=speed_kmh,
                   position=self.depot_coords, status=STATE_IDLE)
        self.trucks[vehicle_id] = t
        return t

    def assign_route(self, vehicle_id: int, node_sequence: list, legs: list):
        """Gán 1 tuyến (đã có road-geometry từng leg) cho 1 xe. node_sequence
        LUÔN bắt đầu và kết thúc bằng depot_index."""
        t = self.trucks[vehicle_id]
        t.legs = legs
        t.leg_idx = 0
        t.progress_m = 0.0
        t.pending_nodes = [n for n in node_sequence if n != self.depot_index]
        t.position = self.depot_coords
        t.needs_reopt = False
        t.status = STATE_MOVING if legs else STATE_IDLE
        self._log(f"🚛 Truck {vehicle_id}: được gán tuyến {len(t.pending_nodes)} điểm "
                   f"({sum(l.length_m for l in legs)/1000:.2f} km).")

    def redirect_to_depot(self, vehicle_id: int, leg_to_depot: RouteLeg):
        """Chèn 1 leg trực tiếp (từ vị trí hiện tại) về depot, dùng khi xe cần
        quay đầu giữa tuyến (đầy tải / hỏng hàng xóm / vượt actual waste)."""
        t = self.trucks[vehicle_id]
        t.legs = [leg_to_depot]
        t.leg_idx = 0
        t.progress_m = 0.0
        t.status = STATE_RETURNING_TO_DEPOT
        self._log(f"↩️ Truck {vehicle_id}: quay về DEPOT để xả tải "
                   f"({leg_to_depot.length_m/1000:.2f} km).")

    # -------------------------------------------------------- update actual
    def set_actual_waste(self, node_index: int, actual_kg: float):
        old = self.demands_actual.get(node_index, self.demands_predicted.get(node_index, 0.0))
        self.demands_actual[node_index] = actual_kg
        self._log(f"📈 Cập nhật lượng rác thực tế tại node {node_index}: "
                   f"{old:.0f} kg → {actual_kg:.0f} kg "
                   f"(dự báo: {self.demands_predicted.get(node_index, 0.0):.0f} kg).")

    def add_extra_node(self, node_index: int, demand_kg: float):
        """Điểm phát sinh thêm rác ngoài dự báo (mục 9 phần yêu cầu)."""
        self.demands_predicted.setdefault(node_index, 0.0)
        self.demands_actual[node_index] = demand_kg
        self.service_times_s.setdefault(node_index, 180.0)
        self.unassigned_pending.append(node_index)
        self._log(f"➕ Điểm phát sinh mới: node {node_index} ({demand_kg:.0f} kg) — chờ tái tối ưu.")

    # ------------------------------------------------------------- events
    def mark_broken(self, vehicle_id: int):
        t = self.trucks.get(vehicle_id)
        if t is None or t.status == STATE_BROKEN:
            return
        remaining = [t.next_node] if t.status in (STATE_MOVING, STATE_COLLECTING) and t.next_node is not None else []
        remaining += [n for n in t.pending_nodes if n not in remaining]
        t.status = STATE_BROKEN
        t.legs = []
        t.pending_nodes = []
        if remaining:
            self.unassigned_pending.extend(remaining)
        self._log(f"🔴 Truck {vehicle_id} BROKEN. {len(remaining)} điểm cần tái phân bổ cho xe khác.")

    def repair_truck(self, vehicle_id: int):
        t = self.trucks.get(vehicle_id)
        if t is None:
            return
        t.status = STATE_IDLE
        t.position = self.depot_coords
        t.current_load_kg = 0.0
        self._log(f"🔧 Truck {vehicle_id} đã sửa xong, trở lại DEPOT sẵn sàng nhận tuyến mới.")

    def block_road(self, node_a: int, node_b: int):
        key = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
        self.blocked_edges.add(key)
        self._log(f"⚠️ Road blockage: đoạn {node_a} ↔ {node_b} bị chặn.")
        for t in self.trucks.values():
            if t.status in (STATE_MOVING, STATE_COLLECTING) and t.leg_idx < len(t.legs):
                leg = t.legs[t.leg_idx]
                if leg.edge_key() == key:
                    t.status = STATE_BLOCKED
                    if leg.to_node not in t.pending_nodes:
                        t.pending_nodes.insert(0, leg.to_node)
                    self._log(f"🚧 Truck {t.vehicle_id} bị chặn đường, cần định tuyến lại.")

    def unblock_road(self, node_a: int, node_b: int):
        key = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
        self.blocked_edges.discard(key)
        self._log(f"✅ Đã gỡ chặn đoạn {node_a} ↔ {node_b}.")

    def is_edge_blocked(self, node_a: int, node_b: int) -> bool:
        key = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
        return key in self.blocked_edges

    # ------------------------------------------------------------ pool cần re-optimize
    def trucks_needing_reopt(self) -> list:
        return [t.vehicle_id for t in self.trucks.values() if t.needs_reopt]

    def idle_trucks(self) -> list:
        return [t.vehicle_id for t in self.trucks.values() if t.status == STATE_IDLE]

    def blocked_trucks(self) -> list:
        return [t.vehicle_id for t in self.trucks.values() if t.status == STATE_BLOCKED]

    def pending_pool(self) -> list:
        """Toàn bộ node hiện chưa được thu gom và không thuộc xe nào đang xử
        lý (dùng để feed vào solver khi tái tối ưu cho các xe đang rảnh)."""
        return list(self.unassigned_pending)

    def clear_pending_pool(self):
        self.unassigned_pending = []

    # -------------------------------------------------------------- tick
    def tick(self, dt_seconds: float):
        """Tiến simulation thêm dt_seconds (giây mô phỏng). Không di chuyển
        xe ngoài khung giờ hoạt động (trừ khi đang COLLECTING/RETURNING dở
        dang — vẫn cho hoàn tất để tránh "đứng hình" giữa đường)."""
        self.clock_min += dt_seconds / 60.0
        in_window = is_within_operating_windows(self.clock_min, self.operating_windows_min)

        for t in list(self.trucks.values()):
            if t.status in (STATE_BROKEN, STATE_BLOCKED, STATE_IDLE, STATE_COMPLETED):
                continue
            if not in_window and t.status == STATE_MOVING and t.progress_m == 0.0:
                continue  # chưa xuất phát và đang ngoài giờ -> chờ

            if t.status == STATE_COLLECTING:
                t.collect_timer_s -= dt_seconds
                t.total_active_time_s += dt_seconds
                if t.collect_timer_s <= 0:
                    self._finish_collecting(t)
                continue

            if t.status in (STATE_MOVING, STATE_RETURNING_TO_DEPOT):
                self._advance_truck(t, dt_seconds)

    def _advance_truck(self, t: Truck, dt_seconds: float):
        if t.leg_idx >= len(t.legs):
            return
        leg = t.legs[t.leg_idx]
        speed_mps = max(0.1, t.speed_kmh) * 1000.0 / 3600.0
        step_m = speed_mps * dt_seconds
        t.progress_m += step_m
        t.total_distance_m += step_m
        t.total_active_time_s += dt_seconds

        if leg.length_m <= 0 or t.progress_m >= leg.length_m:
            # Đến node cuối của leg
            t.position = (
                (leg.geometry[-1][0], leg.geometry[-1][1]) if leg.geometry else t.position
            )
            t.progress_m = 0.0
            arrived_node = leg.to_node
            t.leg_idx += 1

            if arrived_node == self.depot_index:
                self._arrive_depot(t)
            else:
                t.status = STATE_COLLECTING
                t.collect_timer_s = max(30.0, self.service_times_s.get(arrived_node, 180.0))
                self._log(f"🚛 Truck {t.vehicle_id}: đến node {arrived_node}, bắt đầu thu gom.")
        else:
            lat, lon, _ = interpolate_along_route(leg.geometry, t.progress_m)
            if lat is not None:
                t.position = (lat, lon)

    def _finish_collecting(self, t: Truck):
        node = t.legs[t.leg_idx - 1].to_node if t.leg_idx > 0 else None
        if node is None:
            t.status = STATE_MOVING
            return

        predicted = self.demands_predicted.get(node, 0.0)
        actual = self.demands_actual.get(node, predicted)

        if t.current_load_kg + actual > t.capacity_kg + 1e-6:
            # Vượt tải nếu thu điểm này -> BỎ LẠI, quay depot xả tải trước.
            if node not in t.pending_nodes:
                t.pending_nodes.insert(0, node)
            note = "" if abs(actual - predicted) < 1e-6 else f" (thực tế {actual:.0f}kg > dự báo {predicted:.0f}kg)"
            self._log(
                f"🟠 Truck {t.vehicle_id}: node {node} sẽ vượt tải "
                f"({t.current_load_kg:.0f}+{actual:.0f} > {t.capacity_kg:.0f}kg){note} → bỏ lại, quay DEPOT."
            )
            t.status = STATE_FULL
            self._route_remaining_direct_to_depot(t)
            return

        t.current_load_kg += actual
        if node in t.pending_nodes:
            t.pending_nodes.remove(node)
        t.served_nodes.append(node)
        var_note = "" if abs(actual - predicted) < 1e-6 else f" | thực tế {actual:.0f}kg (dự báo {predicted:.0f}kg)"
        self._log(f"✅ Truck {t.vehicle_id}: thu gom {actual:.0f}kg tại node {node}{var_note}.")

        if t.capacity_utilization_pct >= self.full_load_threshold_pct:
            self._log(f"🟠 Truck {t.vehicle_id}: đầy tải ({t.capacity_utilization_pct:.0f}%) → quay DEPOT.")
            t.status = STATE_FULL
            self._route_remaining_direct_to_depot(t)
            return

        # Còn chỗ + còn leg kế tiếp -> tiếp tục di chuyển theo tuyến đã gán.
        if t.leg_idx < len(t.legs):
            t.status = STATE_MOVING
        else:
            # Hết leg trong tuyến hiện có mà chưa quay depot (trường hợp hiếm) -> coi như xong tuyến.
            self._arrive_depot(t)

    def _route_remaining_direct_to_depot(self, t: Truck):
        """Không còn leg OSRM sẵn có để về depot trực tiếp trong prototype
        offline; đánh dấu needs_reopt=True để lớp gọi (app.py) tính leg thật
        (OSRM) từ vị trí hiện tại về depot rồi gọi redirect_to_depot()."""
        t.needs_reopt = True

    def _arrive_depot(self, t: Truck):
        t.current_load_kg = 0.0
        t.position = self.depot_coords
        t.legs = []
        t.leg_idx = 0
        if t.pending_nodes:
            t.status = STATE_IDLE
            t.needs_reopt = True
            self._log(f"🏠 Truck {t.vehicle_id}: đã về DEPOT, xả tải. "
                       f"Còn {len(t.pending_nodes)} điểm → chờ tái tối ưu.")
        else:
            t.status = STATE_COMPLETED
            self._log(f"🎉 Truck {t.vehicle_id}: hoàn thành toàn bộ tuyến được giao.")

    # -------------------------------------------------------------- KPIs
    def total_predicted_waste(self) -> float:
        return sum(v for k, v in self.demands_predicted.items() if k != self.depot_index)

    def total_collected_waste(self) -> float:
        return sum(t.current_load_kg for t in self.trucks.values()) + self._total_unloaded_waste()

    def _total_unloaded_waste(self) -> float:
        # Rác đã xả ở depot: tổng actual của các node đã served TRỪ phần hiện
        # đang còn trên xe (current_load_kg đã tính riêng ở trên).
        served_total = 0.0
        onboard_nodes = set()
        for t in self.trucks.values():
            served_total += sum(self.demands_actual.get(n, self.demands_predicted.get(n, 0.0)) for n in t.served_nodes)
        return served_total

    def dashboard_kpis(self, cost_params: dict | None = None) -> dict:
        cost_params = cost_params or {}
        active = [t for t in self.trucks.values() if t.status != STATE_BROKEN]
        broken = [t for t in self.trucks.values() if t.status == STATE_BROKEN]
        total_distance_km = sum(t.total_distance_m for t in self.trucks.values()) / 1000.0
        total_time_h = sum(t.total_active_time_s for t in self.trucks.values()) / 3600.0
        predicted = self.total_predicted_waste()
        collected = sum(
            self.demands_actual.get(n, self.demands_predicted.get(n, 0.0))
            for t in self.trucks.values() for n in t.served_nodes
        )
        onboard = sum(t.current_load_kg for t in self.trucks.values())
        remaining = max(0.0, predicted - collected)
        avg_util = (
            sum(t.capacity_utilization_pct for t in self.trucks.values() if t.status != STATE_BROKEN) / len(active)
            if active else 0.0
        )
        fuel_cost = total_distance_km * cost_params.get("fuel_cost_per_km", 0.0)
        driver_cost = total_time_h * cost_params.get("driver_cost_per_hour", 0.0)
        maint_cost = total_distance_km * cost_params.get("maintenance_cost_per_km", 0.0)
        other_cost = cost_params.get("other_cost", 0.0)
        total_cost = fuel_cost + driver_cost + maint_cost + other_cost
        service_level = 100.0 * collected / predicted if predicted > 0 else 100.0

        return {
            "clock": format_hhmm(self.clock_min),
            "total_predicted_waste_kg": round(predicted, 1),
            "collected_waste_kg": round(collected, 1),
            "onboard_waste_kg": round(onboard, 1),
            "remaining_waste_kg": round(remaining, 1),
            "num_vehicles": len(self.trucks),
            "active_vehicles": len(active),
            "broken_vehicles": len(broken),
            "total_distance_km": round(total_distance_km, 2),
            "total_time_h": round(total_time_h, 2),
            "avg_utilization_pct": round(avg_util, 1),
            "service_level_pct": round(service_level, 1),
            "fuel_cost": round(fuel_cost, 0),
            "driver_cost": round(driver_cost, 0),
            "maintenance_cost": round(maint_cost, 0),
            "other_cost": round(other_cost, 0),
            "total_cost": round(total_cost, 0),
        }

    def vehicle_table(self) -> list:
        rows = []
        for t in sorted(self.trucks.values(), key=lambda x: x.vehicle_id):
            rows.append({
                "Vehicle": f"Truck {t.vehicle_id:02d}",
                "Status": t.status,
                "Load (kg)": round(t.current_load_kg, 1),
                "Capacity (kg)": round(t.capacity_kg, 1),
                "Utilization (%)": round(t.capacity_utilization_pct, 1),
                "Current Node": t.current_node if t.current_node is not None else "-",
                "Next Node": t.next_node if t.next_node is not None else "-",
                "Distance (km)": round(t.total_distance_m / 1000.0, 2),
                "Remaining Points": len(t.pending_nodes),
            })
        return rows

    def route_table(self, node_label) -> list:
        rows = []
        for t in sorted(self.trucks.values(), key=lambda x: x.vehicle_id):
            seq = [t.current_node] if t.current_node is not None else []
            seq += [n for n in ([t.next_node] if t.next_node is not None else []) + t.pending_nodes]
            labels = " → ".join(dict.fromkeys(node_label(n) for n in seq if n is not None)) or "-"
            rows.append({
                "Vehicle": f"Truck {t.vehicle_id:02d}",
                "Route (còn lại)": labels,
                "Distance so far (km)": round(t.total_distance_m / 1000.0, 2),
                "Waste onboard (kg)": round(t.current_load_kg, 1),
                "Utilization (%)": round(t.capacity_utilization_pct, 1),
                "Status": t.status,
            })
        return rows
