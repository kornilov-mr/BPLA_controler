#include <Servo.h>
#include <Wire.h>
Servo leftServo, midServo, rightServo;
Servo leftFrontMotor, rightFrontMotor, rightBackMotor, leftBackMotor;

void setup() {
  Wire.begin(0x20);
  delay(1000);
  Wire.onReceive(change_servo);
  delay(1000);
  leftServo.attach(2);
  midServo.attach(3);
  rightServo.attach(4);
  leftFrontMotor.attach(5);
  rightFrontMotor.attach(6);
  rightBackMotor.attach(7);
  leftBackMotor.attach(8);
}

void change_servo(){
  int temp[7], i = 0;
  while (Wire.available()){
    temp[i] = Wire.read();
    i++;
    }
    leftServo.write(temp[0]);
    midServo.write(temp[1]);
    rightServo.write(temp[2]);

    leftFrontMotor.writeMicroseconds(temp[3]);
    rightFrontMotor.writeMicroseconds(temp[4]);

    leftBackMotor.writeMicroseconds(temp[5]);
    rightBackMotor.writeMicroseconds(temp[6]);
}
void loop() {
  delay(1);
}
