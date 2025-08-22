#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Пины энкодера
#define ENCODER_CLK 2   // S1 (прерывание)
#define ENCODER_DT 3    // S2
#define ENCODER_SW 4    // KEY (кнопка)

// Пины выхода
#define PIN_7 7
#define PIN_8 8
#define LED 13

// Пин кнопки блокировки управления реле
#define RELAY_LOCK_BUTTON_PIN 10

// Тайминг
unsigned long delayTime = 8500;  // переменная задержка
volatile bool encoderUpdated = false;

enum State { IDLE, STAGE1, STAGE2, STAGE3 };
State currentState = STAGE1;

unsigned long lastMillis = 0;

// Для кнопки энкодера
bool buttonState = HIGH;
unsigned long lastButtonCheck = 0;
bool buttonPressed = false;

// Для энкодера
volatile int lastCLK = HIGH;

void setup() {
  pinMode(PIN_7, OUTPUT);
  pinMode(PIN_8, OUTPUT);
  digitalWrite(PIN_7, LOW);
  digitalWrite(PIN_8, LOW);

  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);
  delay(1000);
  digitalWrite(LED, HIGH);
  delay(1000);
  digitalWrite(LED, LOW);

  pinMode(ENCODER_CLK, INPUT_PULLUP);
  pinMode(ENCODER_DT, INPUT_PULLUP);
  pinMode(ENCODER_SW, INPUT_PULLUP);
  pinMode(RELAY_LOCK_BUTTON_PIN, INPUT_PULLUP);  // Кнопка блокировки реле

  attachInterrupt(digitalPinToInterrupt(ENCODER_CLK), handleEncoder, FALLING);

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    while (true); // ошибка
  }
  display.setRotation(2);
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  updateDisplay();
}

void loop() {
  unsigned long now = millis();

  // Проверка состояния кнопки блокировки реле
  bool relayLockPressed = (digitalRead(RELAY_LOCK_BUTTON_PIN) == LOW);

  if (!relayLockPressed) {
    // Машина состояний работает только при отжатой кнопке
    switch (currentState) {
      case STAGE1:
        digitalWrite(PIN_7, LOW);
        if (now - lastMillis >= 300) {
          digitalWrite(PIN_8, LOW);
          lastMillis = now;
          currentState = STAGE2;
        }
        break;
      case STAGE2:
        if (now - lastMillis >= 700) {
          digitalWrite(PIN_8, HIGH);
          lastMillis = now;
          currentState = STAGE3;
        }
        break;
      case STAGE3:
        if (now - lastMillis >= 100) {
          digitalWrite(PIN_7, HIGH);
          lastMillis = now;
          currentState = IDLE;
        }
        break;
      case IDLE:
        if (now - lastMillis >= delayTime) {
          currentState = STAGE1;
          lastMillis = now;
        }
        break;
    }
  } else {
    // Если кнопка нажата — сбросить реле и состояние автомата
    digitalWrite(PIN_7, HIGH);
    digitalWrite(PIN_8, HIGH);
    currentState = IDLE;  // можно оставить IDLE, чтобы вернуться к работе сразу после отпускания
  }

  // Проверка кнопки энкодера каждые 10 мс
  if (millis() - lastButtonCheck > 10) {
    bool current = digitalRead(ENCODER_SW) == LOW;
    if (current && !buttonState) {
      buttonPressed = true;
    } else if (!current && buttonState) {
      buttonPressed = false;
    }
    buttonState = current;
    lastButtonCheck = millis();
  }

  // Обновление экрана при изменении задержки
  if (encoderUpdated) {
    encoderUpdated = false;
    updateDisplay();
  }
}

void handleEncoder() {
  if (!buttonPressed) return;  // Игнорировать, если кнопка не нажата

  int dtValue = digitalRead(ENCODER_DT);
  if (dtValue == LOW) {
    if (delayTime >= 100) delayTime -= 100;
  } else {
    delayTime += 100;
  }
  encoderUpdated = true;
}

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.print("Delay:");
  display.setCursor(0, 16);
  display.print(delayTime);
  display.print(" ms");
  display.display();
}
