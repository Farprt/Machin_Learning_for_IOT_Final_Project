/*
 * 哈利波特魔杖
 */

#include <Arduino_LSM9DS1.h> 
#include <MicroTFLite.h>     
#include "net.h"            

// --- 1. 参数配置 ---
#define N_INPUTS 561  
#define N_OUTPUTS 16  

constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --- 2. 变量定义 ---
float input_buffer[N_INPUTS];
int buffer_ix = 0;
bool is_recording = false;
const float ACCEL_THRESHOLD = 1.2;

// 简易 KNN 数据库
struct Sample { float features[16]; int label; bool active; };
Sample database[10];
int sample_count = 0;

int train_label = 0;

void setLED(bool r, bool g, bool b) {
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

void setup() {
  Serial.begin(115200);

  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  setLED(false, false, false); 

  delay(3000);

  Serial.println("=== System Initializing ===");
  Serial.flush();

  // 先初始化模型
  Serial.print("1. Initializing Model... ");
  // 在 IMU 启动前，系统是最干净的，这时候分配内存最不容易崩
  if (!ModelInit(g_model, tensor_arena, kTensorArenaSize)) {
    Serial.println("\n❌ Model Init Failed!");
    Serial.println("Hint: Model data might be corrupted or incompatible.");
    while (true) { setLED(0,0,1); delay(100); setLED(0,0,0); delay(100); }
  }
  Serial.println("✅ OK!");
  Serial.flush();
  ModelPrintMetadata();

  Serial.print("2. Initializing IMU... ");

  if (!IMU.begin()) {
    Serial.println("\n❌ IMU Failed!");
    while (1) { setLED(1,0,0); delay(100); setLED(0,0,0); delay(100); }
  }
  Serial.println("✅ OK!");

  Serial.println("--------------------------------");
  Serial.println(">>> 输入 '1' 学习 Flipendo");
  Serial.println(">>> 输入 '2' 学习 Wingardium");
  Serial.println("--------------------------------");
  
  // 绿灯长亮 2 秒
  setLED(0, 1, 0); delay(2000); setLED(0, 0, 0);
}

float gravityX = 0, gravityY = 0, gravityZ = 0;
// 滤波系数 (0.0 < alpha < 1.0)
// alpha 越小：重力更新越慢，去重力越平滑，但对快速转腕的适应慢。
// alpha 越大：重力更新越快，但可能会把慢速动作也当成重力滤掉了。
const float ALPHA = 0.2;

void loop() {
  // --- 串口指令 ---
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') { train_label = 1; Serial.println("\n[LEARN] Flipendo"); setLED(0, 0, 1); }
    if (c == '2') { train_label = 2; Serial.println("\n[LEARN] Wingardium"); setLED(0, 0, 1); }
    // if (c == 'c') { sample_count = 0; for(int i=0; i<10; i++) database[i].active=false; Serial.println("\nCleared"); }
  }

  // --- 数据采集 ---
  if (IMU.accelerationAvailable()) {
    float rawX, rawY, rawZ;
    IMU.readAcceleration(rawX, rawY, rawZ);

    // 1. 低通滤波 (Low-Pass Filter) -> 提取出当前的重力分量
    gravityX = ALPHA * rawX + (1 - ALPHA) * gravityX;
    gravityY = ALPHA * rawY + (1 - ALPHA) * gravityY;
    gravityZ = ALPHA * rawZ + (1 - ALPHA) * gravityZ;

    // 2. 高通滤波 (High-Pass Filter) -> 原始值 - 重力 = 纯动作
    // 这里的 linear_x/y/z 就是去除了重力后的纯净加速度
    float x = rawX - gravityX;
    float y = rawY - gravityY;
    float z = rawZ - gravityZ;

    if (!is_recording && (fabs(x) + fabs(y) + fabs(z)) > ACCEL_THRESHOLD) {
      is_recording = true;
      buffer_ix = 0;
    }

    if (is_recording) {
      input_buffer[buffer_ix++] = x;
      input_buffer[buffer_ix++] = y;
      input_buffer[buffer_ix++] = z;

      if (buffer_ix >= N_INPUTS) {
        is_recording = false;
        setLED(false, false, false); 
        process_gesture();           
        delay(1000);                 
      }
      delay(16); 
    }
  }
}

void process_gesture() {
  // Serial.println("\n--- [DEBUG] Input Matrix Start (CSV Format: x,y,z) ---");
  
  // // input_buffer 是一维数组，排列方式是 [x0, y0, z0, x1, y1, z1, ...]
  // // 我们每 3 个一组打印，还原为时间步
  // for (int i = 0; i < N_INPUTS; i += 3) {
  //   Serial.print(input_buffer[i], 4);     // x，保留4位小数
  //   Serial.print(",");
  //   Serial.print(input_buffer[i+1], 4);   // y
  //   Serial.print(",");
  //   Serial.println(input_buffer[i+2], 4); // z
  // }
  // Serial.println("--- [DEBUG] Input Matrix End ---\n");

  // 1. 填入数据
  for (int i = 0; i < N_INPUTS; i++) {
    if (!ModelSetInput(input_buffer[i], i)) return;
  }

  // 2. 推理
  if (!ModelRunInference()) {
    Serial.println("Inference Failed");
    return;
  }

  // 3. 获取输出
  float embedding[16];
  for (int i = 0; i < 16; i++) embedding[i] = ModelGetOutput(i);

  // 4. KNN 逻辑
  if (train_label != 0) {
    if (sample_count < 10) {
      for(int i=0; i<16; i++) database[sample_count].features[i] = embedding[i];
      database[sample_count].label = train_label;
      database[sample_count].active = true;
      sample_count++;
      Serial.print("✅ Learned #"); Serial.println(sample_count);
      train_label = 0; 
      setLED(0, 1, 0); delay(200); setLED(0, 0, 0); delay(200);
      setLED(0, 1, 0); delay(200); setLED(0, 0, 0);
    } else {
      Serial.println("Full");
    }
  } else {
    if (sample_count == 0) return;
    float min_dist = 10000.0;
    int best = 0;
    for (int i=0; i<10; i++) {
      if (!database[i].active) continue;
      float d = 0;
      for(int j=0; j<16; j++) {
        float diff = embedding[j] - database[i].features[j];
        d += diff * diff;
      }
      if (d < min_dist) { min_dist = d; best = database[i].label; }
    }

    if (min_dist < 20000.0) { 
      if (best == 1) { Serial.println("PLAY_FLIPENDO"); setLED(1, 0, 0); delay(500); setLED(0, 0, 0); }
      if (best == 2) { Serial.println("PLAY_WINGARDIUM"); setLED(0, 1, 0); delay(500); setLED(0, 0, 0); }
    } else {
      Serial.print("Unknown: "); Serial.println(min_dist);
    }
  }
}