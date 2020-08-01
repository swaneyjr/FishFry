int readoutA = 2;
int readoutB = 3;
int readoutC = 18;
int readoutH = 19;

int ledA = 7;
int ledB = 12;
int ledC = 8;
int ledH = 13;

int compareA = 9;
int compareB = 10;
int compareC = 11;

int threshA = 255;
int threshB = 255;
int threshC = 255;

// LED flash length in micros
const unsigned long duration = 300000;

// all of these will flash at startup, but
// that might not be a bad thing
volatile unsigned long latestA = 0;
volatile unsigned long latestB = 0;
volatile unsigned long latestC = 0;
volatile unsigned long latestH = 0;

void setup() {
  
  // configure pins
  pinMode(readoutA, INPUT_PULLUP);
  pinMode(readoutB, INPUT_PULLUP);
  pinMode(readoutC, INPUT_PULLUP);
  pinMode(readoutH, INPUT_PULLUP);

  pinMode(ledA, OUTPUT);
  pinMode(ledB, OUTPUT);
  pinMode(ledC, OUTPUT);
  pinMode(ledH, OUTPUT);

  pinMode(compareA, OUTPUT);
  pinMode(compareB, OUTPUT);
  pinMode(compareC, OUTPUT);

  Serial.begin(115200);

  // led check at start-up:
  digitalWrite(ledA, true);
  delay(500);
  digitalWrite(ledA, false);
  digitalWrite(ledB, true);
  delay(500);
  digitalWrite(ledB, false);
  digitalWrite(ledC, true);
  delay(500);
  digitalWrite(ledC, false);
  digitalWrite(ledH, true);
  delay(500);
  digitalWrite(ledH, false);

  while(!Serial) {
    ; // wait for Serial port to connect
  }

  digitalWrite(ledA, true);
  digitalWrite(ledB, true);
  digitalWrite(ledC, true);
  digitalWrite(ledH, true);
  delay(500);
  digitalWrite(ledA, false);
  digitalWrite(ledB, false);
  digitalWrite(ledC, false);
  digitalWrite(ledH, false);
 
  analogWrite(compareA, 255-threshA);
  analogWrite(compareB, 255-threshB);
  analogWrite(compareC, 255-threshC);

  Serial.println("hodoscope initialized");
  
  attachInterrupt(digitalPinToInterrupt(readoutA), trigA, FALLING);
  attachInterrupt(digitalPinToInterrupt(readoutB), trigB, FALLING);
  attachInterrupt(digitalPinToInterrupt(readoutC), trigC, FALLING);
  attachInterrupt(digitalPinToInterrupt(readoutH), trigH, FALLING);

}

void trigA() {
  latestA = micros();
  Serial.print("A ");
  Serial.println(latestA);
}

void trigB() {
  latestB = micros();
  Serial.print("B ");
  Serial.println(latestB);
}

void trigC() { 
  latestC = micros();
  Serial.print("C ");
  Serial.println(latestC);
}

void trigH() {
  latestH = micros();
  Serial.print("H ");
  Serial.println(latestH);
}

void updateThresh() {
  bool updated = false;
  
  while(Serial.available() > 0) {
    String threshStr = Serial.readStringUntil('\n');
    Serial.print("Input: ");
    Serial.println(threshStr);
    
    String preStr = threshStr.substring(0, 3);
    int newThresh = threshStr.substring(3).toInt();

    // make sure we can actually parse this
    if (!newThresh) continue;
    
    if (preStr == "A: " && newThresh != threshA) {
      threshA = newThresh;
      updated = true;
    } else if (preStr == "B: " && newThresh != threshB) {
      threshB = newThresh;
      updated = true;
    } else if (preStr == "C: " && newThresh != threshC) { 
      threshC = newThresh;
      updated = true;
    }

  }

  if (updated) {

    analogWrite(compareA, 255-threshA);
    analogWrite(compareB, 255-threshB);
    analogWrite(compareC, 255-threshC);
  
    Serial.print("pwm thresholds: ");
    Serial.print(threshA);
    Serial.print(" ");
    Serial.print(threshB);
    Serial.print(" ");
    Serial.println(threshC);
  }
}

void loop() {

  updateThresh();
  
  unsigned long t = micros();
  digitalWrite(ledA, t < latestA + duration);
  digitalWrite(ledB, t < latestB + duration);  
  digitalWrite(ledC, t < latestC + duration);
  digitalWrite(ledH, t < latestH + duration);
}
