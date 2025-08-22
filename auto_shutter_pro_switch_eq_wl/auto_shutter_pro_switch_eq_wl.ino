#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <EEPROM.h>

// ================= OLED =================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// =============== Пины/кнопки ===============
#define ENCODER_CLK 2
#define ENCODER_DT  3
#define ENCODER_SW  4   // кнопка энкодера: без удержания шаг 0.1s, с удержанием шаг 10s

#define PIN_7 7
#define PIN_8 8
#define LED 13

#define RELAY_LOCK_BUTTON_PIN 10
#define MODE_BUTTON 6     // HIGH = Long, LOW = Short

// =============== Тайминги ===============
const unsigned long T_STAGE1_AFTER_7_LOW  = 300;  // полунажатие выдержка
const unsigned long T_STAGE2_HOLD_8_LOW   = 700;  // полное нажатие выдержка
const unsigned long T_STAGE3_AFTER_7_HIGH = 100;  // пауза после отпускания полного
// Базовая накладка на каждый кадр: 0.3 + 0.7 + 0.1 = 1.1s
const unsigned long T_BASE_DELAY = T_STAGE1_AFTER_7_LOW + T_STAGE2_HOLD_8_LOW + T_STAGE3_AFTER_7_HIGH;

const unsigned long T_SAVE_PAUSE = 5000 - T_BASE_DELAY;

const unsigned long SAVE_DELAY_MS = 10000;  // отложенная запись в EEPROM
const unsigned long DEBOUNCE_MS   = 84;     // антидребезг

// допустимые границы пользовательской задержки (без базы)
const unsigned long MIN_DELAY_MS   = 100;           // 0.1 s
const unsigned long MAX_DELAY_MS   = 60UL*60*1000;  // 60 min

// =============== Состояния ===============
enum State { IDLE, STAGE1, STAGE2, STAGE3, STAGE4, STAGE5 };
State currentStateStandard = STAGE1;
State currentStateLong     = STAGE1;

unsigned long lastMillisStandard = 0;
unsigned long lastMillisLong     = 0;

// =============== Параметр задержки (без базы) ===============
volatile unsigned long delayTime = 8500;   // мс
volatile bool encoderUpdated = false;
volatile bool buttonPressed  = false;

// =============== Антидребезг ===============
struct DebouncedButton {
  uint8_t pin;
  unsigned long debounceMs;
  int lastStable;
  int lastRead;
  unsigned long lastEdgeMs;

  DebouncedButton(uint8_t p, unsigned long d=DEBOUNCE_MS)
    : pin(p), debounceMs(d), lastStable(HIGH), lastRead(HIGH), lastEdgeMs(0) {}

  void beginPullup() {
    pinMode(pin, INPUT_PULLUP);
    int lvl = digitalRead(pin);
    lastStable = lastRead = lvl;
    lastEdgeMs = millis();
  }

  void update(unsigned long now) {
    int raw = digitalRead(pin);
    if (raw != lastRead) {
      lastRead = raw;
      lastEdgeMs = now;
    }
    if ((now - lastEdgeMs) >= debounceMs && lastStable != raw) {
      lastStable = raw;
    }
  }

  bool isLow()  const { return lastStable == LOW;  }
  bool isHigh() const { return lastStable == HIGH; }
  int  level()  const { return lastStable; }
};

DebouncedButton swBtn(ENCODER_SW);
DebouncedButton lockBtn(RELAY_LOCK_BUTTON_PIN);
DebouncedButton modeBtn(MODE_BUTTON);

// =============== EEPROM WEAR-LEVELING (простой кольцевой журнал) ===============
const int EE_ADDR  = 0;
const uint8_t EE_MAGIC = 0x5A;
const int EE_SLOTS = 128;  // используем 128 слотов (вся EEPROM Nano)

struct PersistRec {
  uint32_t value;   // delayTime (ms)
  uint16_t seq;     // счётчик записей
  uint8_t  magic;   // валидность записи
};

int ee_next_slot = 0;     // следующий слот для записи
uint16_t ee_seq  = 0;     // текущий seq

inline unsigned long clampDelay(unsigned long v) {
  if (v < MIN_DELAY_MS) return MIN_DELAY_MS;
  if (v > MAX_DELAY_MS) return MAX_DELAY_MS;
  return v;
}

void eepromLoadSimple() {
  bool found = false;
  uint16_t maxSeq = 0;
  PersistRec best = {};

  for (int i = 0; i < EE_SLOTS; ++i) {
    PersistRec r;
    int addr = EE_ADDR + i * (int)sizeof(PersistRec);
    EEPROM.get(addr, r);
    if (r.magic == EE_MAGIC && r.value >= MIN_DELAY_MS && r.value <= MAX_DELAY_MS) {
      if (!found || r.seq > maxSeq) {
        best = r;
        maxSeq = r.seq;
        ee_next_slot = (i + 1) % EE_SLOTS;  // следующий после найденного
        found = true;
      }
    }
  }

  if (found) {
    noInterrupts();
    delayTime = best.value;
    interrupts();
    ee_seq = maxSeq;
  } else {
    // первая инициализация
    PersistRec w;
    noInterrupts();
    w.value = clampDelay(delayTime);
    interrupts();
    w.seq   = 1;
    w.magic = EE_MAGIC;
    EEPROM.put(EE_ADDR, w);
    noInterrupts();
    delayTime = w.value;
    interrupts();
    ee_seq = 1;
    ee_next_slot = 1 % EE_SLOTS;
  }
}


bool eepromDirty = false;
unsigned long lastDelayEditMs = 0;

void eepromSaveIfNeeded(unsigned long now, unsigned long localDelay) {
  if (eepromDirty && (now - lastDelayEditMs) >= SAVE_DELAY_MS) {
    PersistRec w;
    w.value = clampDelay(localDelay);
    w.seq   = ee_seq + 1;
    w.magic = EE_MAGIC;

    int addr = EE_ADDR + ee_next_slot * (int)sizeof(PersistRec);
    EEPROM.put(addr, w);

    ee_seq = w.seq;
    ee_next_slot = (ee_next_slot + 1) % EE_SLOTS;
    eepromDirty = false;
  }
}


// ================= OLED =================
// Short: показываем total = delay + T_BASE_DELAY
// Long:  показываем только raw delay (без базы)
void updateDisplay(bool shortMode) {
  noInterrupts();
  unsigned long localDelay = delayTime;
  interrupts();

  localDelay = clampDelay(localDelay);

  // что показываем на экране
  const unsigned long shownMs = shortMode
                                ? (localDelay + T_BASE_DELAY)  // Short → total
                                : (localDelay);                // Long  → raw

  display.clearDisplay();
  display.setCursor(0, 0);

  if (shortMode) {
    display.print(F("Mode:  Short"));
    display.setCursor(0, 12);
    display.print(F("Delay: "));
  } else {
    display.print(F("Mode:  Long"));
    display.setCursor(0, 12);
    display.print(F("Exp:   "));
  }

  float seconds = shownMs / 1000.0f;
  display.print(seconds, 1);   // одна цифра после запятой
  display.print(F("s"));

  display.setCursor(0, 24);
  if (shortMode) {
    display.print(F("(b. delay is "));
    display.print(T_BASE_DELAY / 1000.0f);
    display.print(F("s)"));
  }

  display.display();
}

// =============== ENCODER ISR ===============
void handleEncoder() {
  // шаг выбираем по состоянию кнопки (стабильное состояние обновляется в loop)
  const unsigned long step = buttonPressed ? 10000UL : 100UL; // 10s или 0.1s

  unsigned long v = delayTime;
  if (digitalRead(ENCODER_DT) == LOW) {
    if (v >= step) v -= step; else v = MIN_DELAY_MS;
  } else {
    v += step;
  }
  v = clampDelay(v);
  delayTime = v;
  encoderUpdated = true;
}

// =============== SETUP ===============
void setup() {
  pinMode(PIN_7, OUTPUT);
  pinMode(PIN_8, OUTPUT);
  digitalWrite(PIN_7, LOW);
  digitalWrite(PIN_8, LOW);

  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);
  delay(300);
  digitalWrite(LED, HIGH);
  delay(300);
  digitalWrite(LED, LOW);

  swBtn.beginPullup();
  lockBtn.beginPullup();
  modeBtn.beginPullup();

  pinMode(ENCODER_CLK, INPUT_PULLUP);
  pinMode(ENCODER_DT,  INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_CLK), handleEncoder, FALLING);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    while (true);
  }
  display.setRotation(0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  eepromLoadSimple();

  unsigned long now = millis();
  lastMillisStandard = now;
  lastMillisLong     = now;

  // LOW = Short → передаём флаг shortMode
  updateDisplay(modeBtn.level() == LOW);
}

// =============== LOOP ===============
void loop() {
  unsigned long now = millis();

  swBtn.update(now);
  lockBtn.update(now);
  modeBtn.update(now);

  // обновляем «стабильное» состояние кнопки энкодера для ISR
  buttonPressed = swBtn.isLow();

  noInterrupts();
  unsigned long localDelay = delayTime;
  bool localEncUpd = encoderUpdated;
  encoderUpdated = false;
  interrupts();

  if (localEncUpd) {
    eepromDirty = true;
    lastDelayEditMs = now;
  }

  bool relayLockActive = lockBtn.isLow();
  // HIGH = Long, LOW = Short
  bool isShortMode = (modeBtn.level() == LOW);

  // безопасный сброс при смене режима
  static bool lastIsShortMode = isShortMode;
  if (isShortMode != lastIsShortMode) {
    digitalWrite(PIN_7, HIGH);
    digitalWrite(PIN_8, HIGH);
    currentStateStandard = STAGE1;
    currentStateLong     = STAGE1;
    lastMillisStandard = now;
    lastMillisLong     = now;
    updateDisplay(isShortMode);
    lastIsShortMode = isShortMode;
  }

  if (!relayLockActive) {
    if (!isShortMode) {
      // -------- LONG: нажатие (без отжатия) → WAIT → отжатие → пауза --------
      switch (currentStateLong) {
        case STAGE1: // полунажатие: 7 LOW
          digitalWrite(PIN_7, LOW);
          if (now - lastMillisLong >= T_STAGE1_AFTER_7_LOW) {
            lastMillisLong = now;
            currentStateLong = STAGE2;
          }
          break;

        case STAGE2: // полное нажатие: 8 LOW (и держим)
          digitalWrite(PIN_8, LOW);
          if (now - lastMillisLong >= T_STAGE2_HOLD_8_LOW) {
            lastMillisLong = now;
            currentStateLong = STAGE3; // экспозиция (удержание обоих)
          }
          break;

        case STAGE3: // удержание экспозиции (7 LOW + 8 LOW)
          if (now - lastMillisLong >= localDelay) {
            lastMillisLong = now;
            currentStateLong = STAGE4; // начинаем «двухступенчатое» отжатие
          }
          break;

        case STAGE4: // отпустить полное: 8 HIGH
          digitalWrite(PIN_8, HIGH);
          if (now - lastMillisLong >= T_STAGE3_AFTER_7_HIGH) {
            lastMillisLong = now;
            currentStateLong = STAGE5;
          }
          break;

        case STAGE5: // отпустить полунажатие: 7 HIGH → пауза
          digitalWrite(PIN_7, HIGH);
          lastMillisLong = now;
          currentStateLong = IDLE;
          break;

        case IDLE:   // пауза сохранения
          if (now - lastMillisLong >= T_SAVE_PAUSE) {
            currentStateLong = STAGE1;
            lastMillisLong = now;
          }
          break;
      }
    } else {
      // ---------- SHORT ----------
      switch (currentStateStandard) {
        case STAGE1: // 7 LOW
          digitalWrite(PIN_7, LOW);
          if (now - lastMillisStandard >= T_STAGE1_AFTER_7_LOW) {
            digitalWrite(PIN_8, LOW);
            lastMillisStandard = now;
            currentStateStandard = STAGE2;
          }
          break;

        case STAGE2: // 8 HIGH (отпустить полное)
          if (now - lastMillisStandard >= T_STAGE2_HOLD_8_LOW) {
            digitalWrite(PIN_8, HIGH);
            lastMillisStandard = now;
            currentStateStandard = STAGE3;
          }
          break;

        case STAGE3: // 7 HIGH (отпустить полунажатие)
          if (now - lastMillisStandard >= T_STAGE3_AFTER_7_HIGH) {
            digitalWrite(PIN_7, HIGH);
            lastMillisStandard = now;
            currentStateStandard = IDLE;
          }
          break;

        case IDLE:   // ожидание периода
          if (now - lastMillisStandard >= localDelay) {
            currentStateStandard = STAGE1;
            lastMillisStandard = now;
          }
          break;
      }
    }
  } else {
    // Блокировка: всё отпустить и обнулить автоматы
    digitalWrite(PIN_7, HIGH);
    digitalWrite(PIN_8, HIGH);
    currentStateStandard = STAGE1;
    currentStateLong     = STAGE1;
    lastMillisStandard = now;
    lastMillisLong     = now;
  }

  // Обновление OLED при изменении задержки
  if (localEncUpd) {
    updateDisplay(isShortMode);
  }

  eepromSaveIfNeeded(now, localDelay);
}
