import serial
import time

FINGER_CLOSE_RANGE = (900, 2200)
PALM_CLOSE_RANGE   = (500, 2500)

class LSCSeriesServo:
    def __init__(self, port, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._healthy = False
        self.connect()
        if self._healthy:
            time.sleep(1.5)  # 硬件稳定期

    def connect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self._healthy = True
            print(f"串口 {self.port} 重连成功")
        except serial.SerialException as e:
            self._healthy = False
            print(f"串口 {self.port} 连接失败: {e}")

    def healthy(self):
        return self._healthy

    def _send_command(self, cmd, params):
        if not self._healthy:
            return False
        length = 1 + len(params)
        buf = bytearray(length + 2)
        buf[0] = buf[1] = 0x55
        buf[2] = length
        buf[3] = cmd
        buf[4:4+len(params)] = params
        checksum = 0
        for i in range(2, len(buf) - 1):
            checksum += buf[i]
        buf[-1] = (~checksum) & 0xFF
        try:
            self.ser.write(buf)
            time.sleep(0.05)          # 硬件消化时间
            return True
        except (serial.SerialException, OSError):
            print("串口写入失败，立即尝试自愈...")
            self._healthy = False
            # 立即重连
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except:
                pass
            time.sleep(0.5)
            self.connect()
            return False

    def move_servos(self, servos, time_ms=500):
        if not servos or not self._healthy:
            return False
        num = len(servos)
        params = bytearray(3 * num + 3)
        params[0] = num
        params[1] = time_ms & 0xFF
        params[2] = (time_ms >> 8) & 0xFF
        idx = 3
        for sid, pos in servos:
            if sid == 6:
                pos = max(PALM_CLOSE_RANGE[0], min(PALM_CLOSE_RANGE[1], pos))
            else:
                pos = max(FINGER_CLOSE_RANGE[0], min(FINGER_CLOSE_RANGE[1], pos))
            params[idx] = sid
            params[idx+1] = pos & 0xFF
            params[idx+2] = (pos >> 8) & 0xFF
            idx += 3
        return self._send_command(0x03, params)

    def close(self):
        self._healthy = False
        if self.ser and self.ser.is_open:
            self.ser.close()