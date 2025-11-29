import threading
import time
from datetime import datetime
from collections import deque

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
import matplotlib.pyplot as plt

from problem2 import ParmSensor

# DB 연결 + 테이블 정의
engine = create_engine("mysql+pymysql://root:1q2w3e4r@127.0.0.1/testdb")
metadata = MetaData()

parm_data_table = Table(
    "parm_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sensor_name", String(32), nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("temperature", Integer, nullable=False),
    Column("illuminance", Integer, nullable=False),
    Column("humidity", Integer, nullable=False),
)


def create_table():
    metadata.create_all(engine)


# insert 함수 (SQLAlchemy Core + engine.begin)
def insert_sensor_data(sensor_name: str, temp: int, light: int, humi: int, created_at):
    stmt = insert(parm_data_table).values(
        sensor_name=sensor_name,
        created_at=created_at,
        temperature=temp,
        illuminance=light,
        humidity=humi,
    )
    with engine.begin() as conn:
        conn.execute(stmt)


# deque 기반 큐
sensorQ = deque()


# 센서 스레드 → 큐에 데이터 push
def sensor_worker(sensor: ParmSensor, interval: int = 10):
    while True:
        sensor.set_data()
        temp, light, humi = sensor.get_data()
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # print(f"{now_str} {sensor.name} — temp {temp}, light {light}, humi {humi}")

        # deque에 push
        sensorQ.append((sensor.name, temp, light, humi, now))

        time.sleep(interval)


# 큐에서 FIFO로 pop → DB 저장
def db_worker(interval: float = 1.0):
    while True:
        while sensorQ:
            sensor_name, temp, light, humi, created_at = sensorQ.popleft()
            print(
                f"{created_at} {sensor_name} — temp {temp}, light {light}, humi {humi}"
            )
            insert_sensor_data(sensor_name, temp, light, humi, created_at)
        time.sleep(interval)


# 테이블에서 데이터 가져오는 get_sensor_data
def get_sensor_data():
    """
    parm_data에서 sensor_name, created_at, temperature만 시간순으로 가져온다.
    """
    stmt = select(
        parm_data_table.c.sensor_name,
        parm_data_table.c.created_at,
        parm_data_table.c.temperature,
    ).order_by(parm_data_table.c.created_at)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return rows


# 센서별 시간(분 단위) 평균 온도 그래프
def plot_avg_temperature_by_sensor():
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    rows = get_sensor_data()

    buckets: dict[tuple[str, datetime], list[int]] = {}

    # (센서, 시간): (온도) 형태로 dict
    for sensor_name, created_at, temp in rows:
        minute_bucket = created_at.replace(second=0, microsecond=0)
        key = (sensor_name, minute_bucket)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(temp)

    # 센서별 (시간, 평균온도) 시퀀스 만들기
    series: dict[str, list[tuple[datetime, float]]] = {}
    for (sensor_name, minute_bucket), temps in buckets.items():
        avg_temp = sum(temps) / len(temps)
        if sensor_name not in series:
            series[sensor_name] = []
        series[sensor_name].append((minute_bucket, avg_temp))

    # 시간순 정렬
    for sensor_name in series:
        series[sensor_name].sort(key=lambda x: x[0])

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    for sensor_name, data in series.items():
        times, temps = [], []
        for t, v in data:
            times.append(t)
            temps.append(v)
        plt.plot(times, temps, marker="o", label=sensor_name)

    plt.title("센서별 분단위 평균 온도")
    plt.xlabel("시간")
    plt.ylabel("평균 온도 (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    create_table()

    sensors = [ParmSensor(f"Parm-{i}") for i in range(1, 6)]

    for sensor in sensors:
        threading.Thread(target=sensor_worker, args=(sensor,), daemon=True).start()

    threading.Thread(target=db_worker, daemon=True).start()

    plot_avg_temperature_by_sensor()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("종료합니다.")
        engine.dispose()


if __name__ == "__main__":
    main()
