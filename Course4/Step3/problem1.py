import random
from datetime import datetime
import time
import threading


class ParmSensor:
    def __init__(self, name) -> None:
        self.name = name
        self.temperature: float
        self.illuminance: float
        self.humidity: float

    def set_data(self):
        self.temperature = random.uniform(20, 30)
        self.illuminance = random.uniform(5000, 10000)
        self.humidity = random.uniform(40, 70)

    def get_data(self) -> tuple[float, float, float]:
        return self.temperature, self.illuminance, self.humidity


def sensor_worker(sensor: ParmSensor, interval: int = 10) -> None:
    while True:
        sensor.set_data()
        temp, light, humi = sensor.get_data()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now_str} {sensor.name} — temp {temp}, light {light}, humi {humi}")
        time.sleep(interval)


def main():
    sensors = [ParmSensor(f"Parm-{i}") for i in range(1, 6)]

    for sensor in sensors:
        t = threading.Thread(
            target=sensor_worker,
            args=(sensor, 10),
            daemon=True,
        )
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("종료합니다.")


if __name__ == "__main__":
    main()
