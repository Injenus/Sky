void setup() {
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
  pinMode(13, OUTPUT);
  digitalWrite(13, 0);
  delay(1000);
  digitalWrite(13, 1);
  delay(1000);
  digitalWrite(13, 0);
  
}

void loop() {
  digitalWrite(7, LOW);   // Включить пин 7
  delay(300);              // Ждать 0.3 секунды
  digitalWrite(8, LOW);   // Включить пин 8
  delay(700);              // Ждать 0.7 секунды
  digitalWrite(8, HIGH);    // Выключить пин 8
  delay(100);
  digitalWrite(7, HIGH);    // Выключить пин 7

  delay(8500); // пауза, ждем

  // 4.4+1.1 = 5.5 + выдержка + (задежрка записи) 
  // задержка записи стремится к нулю если пауза >> выдержки
  // x.y +1.1 = x+1.y+1 + exposure (0.4)
}
