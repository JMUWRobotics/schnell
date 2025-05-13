#include <Arduino.h>
#include "BasicStepperDriver.h"

// motor : https://www.omc-stepperonline.com/de/p-series-ip67-wasserdicht-nema-23-schrittmotor-5-0a-1-8nm-254-95oz-in-23ip67-20
// stepper controller: https://www.omc-stepperonline.com/de/digitaler-schrittmotortreiber-1-0-4-2a-20-50vdc-fuer-nema-17-23-24-schrittmotor-dm542t

// the motor has a 1.8 deg step so 200 steps
#define MOTOR_STEPS 200

// set a safty RPM
#define SAFTY_RPM 600

// step resolution on stepper controller set to 800 step/rev
#define MICROSTEPS 4

// All the wires needed for full functionality
#define DIR 4
#define STEP 3
#define ENABLE 2

//Uncomment line to use enable/disable functionality
//#define SLEEP 13

// set all value to default 
float rpmVal = 60.0;  
float timeVal = 0.0;  
int angleVal = 0; 
int start_delay = 0;

unsigned long start_time = 0;

// start bsic driver
BasicStepperDriver stepper(MOTOR_STEPS, DIR, STEP, ENABLE);

void setup() {

   // enable serial
   Serial.begin(9600);
   while (!Serial) {;}  // Wait for serial port to connect
   Serial.println("Found serial connection of wavegen");

   stepper.begin(rpmVal, MICROSTEPS);
   stepper.setEnableActiveState(LOW);

   stepper.disable();
  
}

void loop(){

   // parse the message from python
   if (Serial.available() > 0) {

      // read in line 
      String input = Serial.readStringUntil('\n');
      input.trim();

      // Split command and value
      int spaceIndex = input.indexOf(' ');
      String command = input;
      int value = 0;

      // Split string into tokens
      const int maxTokens = 10;
      String tokens[maxTokens];
      int tokenCount = 0;
      
      // sort the messages
      while (input.length() > 0 && tokenCount < maxTokens) {
         int spaceIndex = input.indexOf(' ');
         if (spaceIndex == -1) {

            tokens[tokenCount++] = input;
            break;
         
         } else {
         
            tokens[tokenCount++] = input.substring(0, spaceIndex);
            input = input.substring(spaceIndex + 1);
            input.trim();
         
         }
      }

      // Parse name/value pairs
      for (int i = 0; i < tokenCount - 1; i += 2) {
      String name = tokens[i];
      float value = tokens[i + 1].toFloat();

         if (name == "a") {
            angleVal = tokens[i + 1].toInt();
            Serial.print("Set angle to ");
            Serial.println(angleVal);

         } else if (name == "t") {
            
            // time in ms
            timeVal = value;
            Serial.print("Set time to ");
            Serial.println(timeVal);
            
         } else if (name == "rpm") {
            
            if(value > SAFTY_RPM){
               rpmVal = SAFTY_RPM;
            }
            else if(value != 0.0){
               rpmVal = value;
            }
            
            // set the new rpm value
            stepper.begin(rpmVal, MICROSTEPS);

            Serial.print("Set rpm to ");
            Serial.println(rpmVal);

         } else if(name == "delay"){
          
            start_delay = value;
            Serial.print("Delay before start: ");
            Serial.println(start_delay);
  
         } else {
            
            Serial.print("Unknown parameter: ");
            Serial.println(name);
         }
      }

      delay(start_delay * 1000); // in sec
      Serial.println("Starting");      
   
   }
   

   if(angleVal != 0){
      Serial.print("start rotation with angle and rpm: ");
      Serial.print(angleVal);
      Serial.print(" ");
      Serial.println(rpmVal);
      
      stepper.enable();
      // if no rpm given the rotation speed is set to default
      stepper.rotate(angleVal); 

      stepper.disable();
      angleVal = 0;
      rpmVal = 60.0;

      Serial.println("Finished");

   }

   // only set in rpm if the angle is 0 -> else do angular rotation
   if(rpmVal != 60.0 && angleVal == 0){

      Serial.print("rotating at rpm ");
      Serial.println(rpmVal);
      
      // set the new rpm value
      stepper.setRPM(rpmVal);
      
      if(start_time = 0){
        start_time = millis();    
      }
      
      stepper.rotate(360);
   }

   if(timeVal != 0.0){
      //Serial.println(millis() - start_time);
      if(millis() - start_time > timeVal){
         // set RPM and set time to 0
         timeVal = 0.0;
         rpmVal = 60.0;
         stepper.disable();

         Serial.println("Stopped motor due to time");
         Serial.println("Finished");
      }
   }

}
