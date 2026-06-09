#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

#define SERVO_MIN 102
#define SERVO_MAX 800

// Gripper limits
#define GRIPPER_OPEN   420
#define GRIPPER_CLOSED 290

// Base limits
#define BASE_MIN 102
#define BASE_MAX 600

// Joystick 1 pins
#define JOY1_X  A0  // Base horizontal
#define JOY1_Y  A1  // Shoulder vertical

// Joystick 2 pins
#define JOY2_X  A2  // Wrist vertical
#define JOY2_Y  A3  // Elbow vertical

// Servo channels
#define BASE       0
#define SHOULDER   1
#define ELBOW      2
#define WRIST_V    5
#define GRIPPER    6

#define DEADZONE       50
#define SPEED          1
#define MOVE_THRESHOLD 5
#define GRIPPER_STEP_DELAY 1

int posBase     = 450;
int posShoulder = 450;
int posElbow    = 360;
int posWristV   = 360;

int baseCount     = 0;
int shoulderCount = 0;
int elbowCount    = 0;
int wristVCount   = 0;

int  gripperPos    = GRIPPER_CLOSED;
int  gripperTarget = GRIPPER_CLOSED;
bool gripperClosed = true;
unsigned long lastGripperMove = 0;

void setServo(int channel, int pos) {
  pos = constrain(pos, SERVO_MIN, SERVO_MAX);
  pca.setPWM(channel, 0, pos);
}

void updateGripper() {
  if (gripperPos == gripperTarget) return;
  unsigned long now = millis();
  if (now - lastGripperMove < GRIPPER_STEP_DELAY) return;
  lastGripperMove = now;
  if (gripperPos < gripperTarget) {
    gripperPos++;
  } else {
    gripperPos--;
  }
  setServo(GRIPPER, gripperPos);
}

void setup() {
  Serial.begin(9600);
  delay(2000);
  Serial.println("Starting...");

  Wire.begin();
  pca.begin();

  Wire.beginTransmission(0x40);
  byte error = Wire.endTransmission();
  if (error == 0) {
    Serial.println("PCA9685 found!");
  } else {
    Serial.println("PCA9685 NOT found - check wiring!");
  }

  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(50);

  setServo(BASE,     posBase);
  setServo(SHOULDER, posShoulder);
  setServo(ELBOW,    posElbow);
  setServo(WRIST_V,  posWristV);

  gripperPos    = GRIPPER_CLOSED;
  gripperTarget = GRIPPER_CLOSED;
  delay(500);

  Serial.println("Ready!");
}

void loop() {
  // --- Joystick 1: Base and Shoulder ---
  int joy1X = analogRead(JOY1_X) - 512;
  int joy1Y = analogRead(JOY1_Y) - 512;

  if (abs(joy1X) < DEADZONE) joy1X = 0;
  if (abs(joy1Y) < DEADZONE) joy1Y = 0;

  if (joy1X != 0) {
    baseCount++;
    if (baseCount >= MOVE_THRESHOLD) {
      posBase += (joy1X > 0) ? SPEED : -SPEED;
      posBase = constrain(posBase, BASE_MIN, BASE_MAX);
      setServo(BASE, posBase);
    }
  } else {
    baseCount = 0;
  }

  if (joy1Y != 0) {
    shoulderCount++;
    if (shoulderCount >= MOVE_THRESHOLD) {
      posShoulder += (joy1Y > 0) ? SPEED : -SPEED;
      posShoulder = constrain(posShoulder, 190, 443);
      setServo(SHOULDER, posShoulder);
    }
  } else {
    shoulderCount = 0;
  }

  // --- Joystick 2: Elbow and Wrist Vertical ---
  int joy2X = analogRead(JOY2_X) - 512;
  int joy2Y = analogRead(JOY2_Y) - 512;

  if (abs(joy2X) < DEADZONE) joy2X = 0;
  if (abs(joy2Y) < DEADZONE) joy2Y = 0;

  if (joy2Y != 0) {
    elbowCount++;
    if (elbowCount >= MOVE_THRESHOLD) {
      posElbow += (joy2Y > 0) ? SPEED : -SPEED;
      posElbow = constrain(posElbow, 350, 560);
      setServo(ELBOW, posElbow);
    }
  } else {
    elbowCount = 0;
  }

  if (joy2X != 0) {
    wristVCount++;
    if (wristVCount >= MOVE_THRESHOLD) {
      posWristV += (joy2X > 0) ? SPEED : -SPEED;
      posWristV = constrain(posWristV, 150, 500);
      setServo(WRIST_V, posWristV);

    }
  } else {
    wristVCount = 0;
  }

  // --- Serial hand tracking ---
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') return;
    Serial.print("Received: ");
    Serial.println(cmd);
    if (cmd == 'C' && !gripperClosed) {
      gripperClosed = true;
      gripperTarget = GRIPPER_CLOSED;
    } else if (cmd == 'O' && gripperClosed) {
      gripperClosed = false;
      gripperTarget = GRIPPER_OPEN;
    }
  }

  // --- Update gripper ---
  updateGripper();

  delay(5);
}