#include <Arduino_LSM9DS1.h>

// --- 1. 全局变量：用于存储重力分量 ---
float gravityX = 0, gravityY = 0, gravityZ = 0;

// 滤波系数 (0.0 < alpha < 1.0)
// 0.2 是个不错的平衡点。如果你觉得归零太慢，可以改大一点（比如 0.3）；
// 如果觉得归零后数值跳动太大，可以改小一点（比如 0.1）。
const float ALPHA = 0.2; 

void setup() {
  Serial.begin(115200);
  while (!Serial); // 等待串口连接

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // 打印表头
  // LinX/Y/Z = Linear Acceleration (线性加速度，即去重力后的加速度)
  // Activity = 动作幅度 (绝对值之和)
  Serial.println("LinX,LinY,LinZ,Activity"); 
}

void loop() {
  // 必须使用临时变量读取原始值
  float rawX, rawY, rawZ;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(rawX, rawY, rawZ);

    // --- 🌟 核心滤波逻辑开始 ---

    // 1. 低通滤波：更新当前的重力背景
    // 这行代码的意思是：重力 = 旧重力 * 0.8 + 新读数 * 0.2
    // 它会慢慢地“适应”当前的姿态
    gravityX = ALPHA * rawX + (1 - ALPHA) * gravityX;
    gravityY = ALPHA * rawY + (1 - ALPHA) * gravityY;
    gravityZ = ALPHA * rawZ + (1 - ALPHA) * gravityZ;

    // 2. 高通滤波：计算纯动作加速度
    // 原始读数 - 重力背景 = 你的手部动作
    float linX = rawX - gravityX;
    float linY = rawY - gravityY;
    float linZ = rawZ - gravityZ;

    // --- 核心滤波逻辑结束 ---

    // 计算用于触发阈值的 "动作幅度"
    float activity_sum = fabs(linX) + fabs(linY) + fabs(linZ);

    // --- 打印数据到串口绘图仪 ---
    Serial.print(linX);
    Serial.print(",");
    Serial.print(linY);
    Serial.print(",");
    Serial.print(linZ);
    Serial.print(",");
    Serial.println(activity_sum); 
  }
  
  delay(50);
}