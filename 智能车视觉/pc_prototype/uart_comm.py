"""
UART 通信协议模块
实现 $V / $S / $R 帧的构造、校验和发送
serial 库为可选依赖：有则用真实串口，无则用虚拟模式（打印指令）
"""

try:
    import serial
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False


def xor_checksum(frame: str) -> str:
    """计算 $ 到 * 之间的 XOR 校验，返回两位大写十六进制"""
    xor = 0
    for c in frame[1:]:
        xor ^= ord(c)
    return f"{xor:02X}"


def build_v_command(vx: float, vy: float, wz: float, with_checksum: bool = True) -> bytes:
    """构造速度指令 $V,vx,vy,wz*XX\\n"""
    frame = f"$V,{vx:.3f},{vy:.3f},{wz:.3f}"
    if with_checksum:
        frame += f"*{xor_checksum(frame)}"
    return (frame + "\n").encode()


def build_s_command(with_checksum: bool = True) -> bytes:
    """构造紧急停车指令 $S*XX\\n"""
    frame = "$S"
    if with_checksum:
        frame += f"*{xor_checksum(frame)}"
    return (frame + "\n").encode()


def build_r_command(x: float, y: float, yaw: float, with_checksum: bool = True) -> bytes:
    """构造位姿校正指令 $R,x,y,yaw*XX\\n"""
    frame = f"$R,{x:.3f},{y:.3f},{yaw:.3f}"
    if with_checksum:
        frame += f"*{xor_checksum(frame)}"
    return (frame + "\n").encode()


def parse_p_frame(data: str) -> dict | None:
    """解析 MCU 上行的 $P 位姿帧"""
    data = data.strip()
    if not data.startswith("$P,"):
        return None
    parts = data.replace("*", ",").split(",")
    if len(parts) < 7:
        return None
    return {
        "x": float(parts[1]),
        "y": float(parts[2]),
        "yaw": float(parts[3]),
        "vx": float(parts[4]),
        "vy": float(parts[5]),
        "wz": float(parts[6]),
    }


def parse_e_frame(data: str) -> str | None:
    """解析堵转报警，返回报警类型或 None"""
    data = data.strip()
    if data.startswith("$E,"):
        return data.split(",")[1].split("*")[0]
    return None


class VirtualUART:
    """虚拟串口：打印指令到控制台，用于 PC 端无硬件调试"""

    def open(self):
        print("[VirtualUART] opened")

    def close(self):
        print("[VirtualUART] closed")

    def send(self, data: bytes):
        print(f"[VirtualUART] TX: {data.decode().strip()}")

    def recv(self) -> list[str]:
        return []

    def send_velocity(self, vx: float, vy: float, wz: float):
        self.send(build_v_command(vx, vy, wz))

    def send_stop(self):
        self.send(build_s_command())

    def send_pose_correction(self, x: float, y: float, yaw: float):
        self.send(build_r_command(x, y, yaw))


class RealUART:
    """真实串口通信"""

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.02)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send(self, data: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(data)

    def recv(self) -> list[str]:
        if not self.ser or not self.ser.is_open:
            return []
        lines = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
        return lines

    def send_velocity(self, vx: float, vy: float, wz: float):
        self.send(build_v_command(vx, vy, wz))

    def send_stop(self):
        self.send(build_s_command())

    def send_pose_correction(self, x: float, y: float, yaw: float):
        self.send(build_r_command(x, y, yaw))


def create_uart(port: str = "COM3", virtual: bool = False):
    """创建串口实例，virtual=True 或 serial 不可用时使用虚拟模式"""
    if virtual or not _HAS_SERIAL:
        return VirtualUART()
    return RealUART(port)
