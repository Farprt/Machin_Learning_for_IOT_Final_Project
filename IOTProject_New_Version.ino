/* * PROJECT: Magic Wand - Academic Refactored (Custom Serial Feedback)
 * ARCHITECTURE: Static MLP 561 -> 32 -> 16 -> 3
 * DESCRIPTION: On-device learning with experience replay and custom log formatting.
 */

#include <Arduino_LSM9DS1.h>
#include <math.h>

// ===================== System Configurations =====================
#define SAMPLE_LEN      187                         // Samples per gesture window
#define AXIS_COUNT      3                           // X, Y, Z axes
#define DATA_DIM        (SAMPLE_LEN * AXIS_COUNT)   // Input dimension: 561
#define CLASS_TOTAL     3                           // Classes: Others (0), Flipendo (1), Wingardium (2)

// Neural network layer dimensions
#define NEURON_H1       32
#define NEURON_H2       16

// Training hyperparameters
float step_rate     = 0.01f;                        // Learning rate
int   cycle_count   = 6;                            // Training epochs per sample
int   replay_depth  = 3;                            // Experience replay cycles per step
float weight_decay  = 0.0001f;                      // L2 regularization penalty

// Motion detection settings
const float TRG_SENSE    = 1.4f;                    // Motion activation threshold
const float FILTER_ALPHA = 0.2f;                    // Low-pass filter for gravity removal
const int   TICK_MS      = 16;                      // 16ms sampling interval (~62.5Hz)

#define MEM_BANK_SIZE   6                           // Samples stored per class for replay

// ===================== Buffers and Weights =====================
float raw_stream[DATA_DIM];
int   stream_ptr = 0;
bool  is_capturing = false;
float gravity_x = 0, gravity_y = 0, gravity_z = 0;

// Memory pool for Experience Replay
float memory_pool[CLASS_TOTAL][MEM_BANK_SIZE][DATA_DIM];
int   memory_count[CLASS_TOTAL] = {0, 0, 0};
int   memory_ptr[CLASS_TOTAL]   = {0, 0, 0};

int   current_task = -1;                            // -1: Test mode, 0/1/2: Learn mode

// Model parameters (Static allocation)
float wt_l1[NEURON_H1][DATA_DIM], bias_l1[NEURON_H1];
float wt_l2[NEURON_H2][NEURON_H1],  bias_l2[NEURON_H2];
float wt_l3[CLASS_TOTAL][NEURON_H2],bias_l3[CLASS_TOTAL];

// Runtime caches
float current_input[DATA_DIM];
float sum_v1[NEURON_H1], active_v1[NEURON_H1];
float sum_v2[NEURON_H2], active_v2[NEURON_H2];
float out_scores[CLASS_TOTAL];
float out_probs[CLASS_TOTAL];

// ===================== Helper Functions =====================
void updateLED(bool r, bool g, bool b) {
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

void scale_signal(float *vec) {
  const float cap = 4.0f;
  for (int i = 0; i < DATA_DIM; i++) {
    // Clip and normalize sensor data to [-1, 1]
    if (vec[i] > cap)  vec[i] = cap;
    if (vec[i] < -cap) vec[i] = -cap;
    vec[i] /= cap;
  }
}

void build_network() {
  randomSeed(analogRead(A0) + millis());
  srand((unsigned)millis());
  auto initialize = [](float *w, int r, int c, float *b) {
    float limit = sqrtf(6.0f / (r + c));            // Xavier initialization
    for (int i = 0; i < r * c; i++) w[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    for (int i = 0; i < r; i++) b[i] = 0.0f;
  };
  initialize(&wt_l1[0][0], NEURON_H1, DATA_DIM, bias_l1);
  initialize(&wt_l2[0][0], NEURON_H2, NEURON_H1, bias_l2);
  initialize(&wt_l3[0][0], CLASS_TOTAL, NEURON_H2, bias_l3);
  for (int i = 0; i < CLASS_TOTAL; i++) { memory_count[i] = 0; memory_ptr[i] = 0; }
}

// ===================== Training and Inference =====================
void run_inference(const float *data) {
  for (int i = 0; i < DATA_DIM; i++) current_input[i] = data[i];
  // Layer 1
  for (int i = 0; i < NEURON_H1; i++) {
    float sum = bias_l1[i];
    for (int j = 0; j < DATA_DIM; j++) sum += wt_l1[i][j] * current_input[j];
    sum_v1[i] = sum; active_v1[i] = sum > 0 ? sum : 0; // ReLU
  }
  // Layer 2
  for (int i = 0; i < NEURON_H2; i++) {
    float sum = bias_l2[i];
    for (int j = 0; j < NEURON_H1; j++) sum += wt_l2[i][j] * active_v1[j];
    sum_v2[i] = sum; active_v2[i] = sum > 0 ? sum : 0; // ReLU
  }
  // Layer 3 (Softmax preparation)
  float top = -1e10;
  for (int i = 0; i < CLASS_TOTAL; i++) {
    float sum = bias_l3[i];
    for (int j = 0; j < NEURON_H2; j++) sum += wt_l3[i][j] * active_v2[j];
    out_scores[i] = sum;
    if (sum > top) top = sum;
  }
  // Numerical stability for Softmax
  float sum_e = 0.0f;
  for (int i = 0; i < CLASS_TOTAL; i++) {
    out_probs[i] = expf(out_scores[i] - top);
    sum_e += out_probs[i];
  }
  for (int i = 0; i < CLASS_TOTAL; i++) out_probs[i] /= (sum_e + 1e-10f);
}

void run_backprop(const float *x, int label) {
  run_inference(x);
  float grad_z3[CLASS_TOTAL];
  for (int i = 0; i < CLASS_TOTAL; i++) grad_z3[i] = out_probs[i] - (i == label ? 1.0f : 0.0f);
  // Update Layer 3
  float delta_a2[NEURON_H2] = {0};
  for (int i = 0; i < CLASS_TOTAL; i++) {
    bias_l3[i] -= step_rate * grad_z3[i];
    for (int j = 0; j < NEURON_H2; j++) {
      delta_a2[j] += wt_l3[i][j] * grad_z3[i];
      wt_l3[i][j] -= step_rate * (grad_z3[i] * active_v2[j] + weight_decay * wt_l3[i][j]);
    }
  }
  // Update Layer 2
  float delta_a1[NEURON_H1] = {0};
  for (int i = 0; i < NEURON_H2; i++) {
    float grad_z2 = delta_a2[i] * (sum_v2[i] > 0 ? 1.0f : 0.0f);
    bias_l2[i] -= step_rate * grad_z2;
    for (int j = 0; j < NEURON_H1; j++) {
      delta_a1[j] += wt_l2[i][j] * grad_z2;
      wt_l2[i][j] -= step_rate * (grad_z2 * active_v1[j] + weight_decay * wt_l2[i][j]);
    }
  }
  // Update Layer 1
  for (int i = 0; i < NEURON_H1; i++) {
    float grad_z1 = delta_a1[i] * (sum_v1[i] > 0 ? 1.0f : 0.0f);
    bias_l1[i] -= step_rate * grad_z1;
    for (int j = 0; j < DATA_DIM; j++) {
      wt_l1[i][j] -= step_rate * (grad_z1 * current_input[j] + weight_decay * wt_l1[i][j]);
    }
  }
}

// ===================== Core Processing (Matched Serial Output) =====================
void process_trigger() {
  float ready_vec[DATA_DIM];
  float x_mx = 0.0f;

  for (int i = 0; i < DATA_DIM; i++) {
    ready_vec[i] = raw_stream[i];
    // Calculate max absolute value
    float abs_val = fabs(ready_vec[i]);
    if (abs_val > x_mx) x_mx = abs_val;
  }

  scale_signal(ready_vec);
  run_inference(ready_vec);
  int best_match = 0;
  for (int i = 1; i < CLASS_TOTAL; i++) if (out_probs[i] > out_probs[best_match]) best_match = i;

  // Print trigger and metrics
  Serial.println("TRIGGER!");
  Serial.print("x_mx="); Serial.println(x_mx, 6);

  if (current_task >= 0) {
    // Update memory pool for experience replay
    for (int i = 0; i < DATA_DIM; i++) memory_pool[current_task][memory_ptr[current_task]][i] = ready_vec[i];
    memory_ptr[current_task] = (memory_ptr[current_task] + 1) % MEM_BANK_SIZE;
    if (memory_count[current_task] < MEM_BANK_SIZE) memory_count[current_task]++;

    // Learning iterations
    for (int i = 0; i < cycle_count; i++) run_backprop(ready_vec, current_task);
    // Experience replay loop
    for (int c = 0; c < CLASS_TOTAL; c++) {
      if (c == current_task || memory_count[c] == 0) continue;
      for (int r = 0; r < replay_depth; r++) run_backprop(memory_pool[c][rand() % memory_count[c]], c);
    }

    // Training feedback
    Serial.print("[TRAIN] true="); Serial.print(current_task);
    Serial.print(" pred="); Serial.print(best_match);
    Serial.print(" mem=(");
    Serial.print(memory_count[0]); Serial.print(", ");
    Serial.print(memory_count[1]); Serial.print(", ");
    Serial.print(memory_count[2]); Serial.print(") ");
    Serial.print("probs=(");
    Serial.print(out_probs[0], 3); Serial.print(", ");
    Serial.print(out_probs[1], 3); Serial.print(", ");
    Serial.print(out_probs[2], 3); Serial.println(")");

    // Action feedback
    if (current_task == 0) Serial.println("[LEARN] Others");
    else if (current_task == 1) Serial.println("[LEARN] Flipendo");
    else if (current_task == 2) Serial.println("[LEARN] Wingardium");

    updateLED(false, true, false); delay(150); updateLED(false, false, false);
  } else {
    // Prediction feedback
    Serial.print("[PRED] pred="); Serial.print(best_match);
    Serial.print(" probs=(");
    Serial.print(out_probs[0], 3); Serial.print(", ");
    Serial.print(out_probs[1], 3); Serial.print(", ");
    Serial.print(out_probs[2], 3); Serial.println(")");

    if (best_match == 1)      { Serial.println("PLAY_FLIPENDO"); updateLED(true, false, false); delay(350); }
    else if (best_match == 2) { Serial.println("PLAY_WINGARDIUM"); updateLED(false, true, false); delay(350); }
    else                     { Serial.println("OTHERS"); updateLED(false, false, true); delay(350); }
    updateLED(false, false, false);
  }
}

// ===================== Arduino Interface =====================
void setup() {
  Serial.begin(115200);
  delay(1200);
  build_network();
  if (!IMU.begin()) while (1); // Halt if IMU fails
  updateLED(false, true, false); delay(600); updateLED(false, false, false);
}

void loop() {
  if (Serial.available()) {
    char key = Serial.read();
    if (key >= '0' && key <= '2') { 
      current_task = key - '0'; 
      updateLED(false, false, true); 
    }
    if (key == 't') { 
      current_task = -1; 
      Serial.println("[TEST] mode"); // Match screenshot
      updateLED(false, false, false); 
    }
    if (key == 'r') { build_network(); current_task = -1; }
  }

  if (IMU.accelerationAvailable()) {
    float ax, ay, az;
    IMU.readAcceleration(ax, ay, az);
    // Remove gravity
    gravity_x = FILTER_ALPHA * ax + (1.0f - FILTER_ALPHA) * gravity_x;
    gravity_y = FILTER_ALPHA * ay + (1.0f - FILTER_ALPHA) * gravity_y;
    gravity_z = FILTER_ALPHA * az + (1.0f - FILTER_ALPHA) * gravity_z;

    float dx = ax - gravity_x, dy = ay - gravity_y, dz = az - gravity_z;

    // Detection logic
    if (!is_capturing && (fabs(dx) + fabs(dy) + fabs(dz)) > TRG_SENSE) {
      is_capturing = true;
      stream_ptr = 0;
    }

    // Capture window
    if (is_capturing) {
      if (stream_ptr <= DATA_DIM - 3) {
        raw_stream[stream_ptr++] = dx;
        raw_stream[stream_ptr++] = dy;
        raw_stream[stream_ptr++] = dz;
      }
      if (stream_ptr >= DATA_DIM) {
        is_capturing = false;
        process_trigger();
      }
      delay(TICK_MS);
    }
  }
}