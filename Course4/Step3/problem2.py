import random
from datetime import datetime
import time
import threading
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)


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


def insert_sensor_data(sensor_name: str, temp: int, light: int, humi: int):
    stmt = insert(parm_data_table).values(
        sensor_name=sensor_name,
        created_at=datetime.now(),
        temperature=temp,
        illuminance=light,
        humidity=humi,
    )
    with engine.begin() as conn:
        conn.execute(stmt)


class ParmSensor:
    def __init__(self, name) -> None:
        self.name = name
        self.temperature: int
        self.illuminance: int
        self.humidity: int

    def set_data(self):
        self.temperature = random.randint(20, 30)
        self.illuminance = random.randint(5000, 10000)
        self.humidity = random.randint(40, 70)

    def get_data(self) -> tuple[int, int, int]:
        return self.temperature, self.illuminance, self.humidity


def sensor_worker(sensor: ParmSensor, interval: int = 10):
    while True:
        sensor.set_data()
        temp, light, humi = sensor.get_data()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now_str} {sensor.name} — temp {temp}, light {light}, humi {humi}")

        insert_sensor_data(sensor.name, temp, light, humi)

        time.sleep(interval)


def main():
    create_table()

    sensors = [ParmSensor(f"Parm-{i}") for i in range(1, 6)]

    for sensor in sensors:
        t = threading.Thread(target=sensor_worker, args=(sensor,), daemon=True)
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("종료합니다.")
        engine.dispose()


if __name__ == "__main__":
    main()
