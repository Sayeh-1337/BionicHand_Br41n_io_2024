/*
Designed and Developed by Muhammad El-Sayeh for the Bionic Hand Project
Project aim is to control orhtisis and prosthetic hands using EEG signals
this code for Br41n.io 2024 Hackathon
*/

/*
 * This ESP32 code is created by esp32io.com
 *
 * This ESP32 code is released in the public domain
 *
 * For more detail (instruction and wiring diagram), visit https://esp32io.com/tutorials/esp32-servo-motor
 */

/*
When running it on Wokwi for VSCode, you can connect to the 
  simulated ESP32 server by opening http://localhost:8180
  in your browser. This is configured by wokwi.toml. :)
*/

#include <ESP32Servo.h>

#include <WiFi.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
#include <WebServer.h>
#include <uri/UriBraces.h>

#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""
// Defining the WiFi channel speeds up the connection:
#define WIFI_CHANNEL 6

//Defining the servo motor pins
#define THUMB_PIN 14 // ESP32 pin GPIO26 connected to servo motor
#define INDEX_PIN 27 // ESP32 pin GPIO25 connected to servo motor
#define MIDDLE_PIN 26 // ESP32 pin GPIO33 connected to servo motor
#define RING_PIN 25 // ESP32 pin GPIO32 connected to servo motor
#define LITTLE_PIN 33 // ESP32 pin GPIO35 connected to servo motor
#define THUMB_A_PIN 32 // ESP32 pin GPIO26 connected to servo motor

// Creating an instance of the WebServer
WebServer server(80);

//thumb index middle ring  little
Servo thumb;
Servo index_f;
Servo middle;
Servo ring;
Servo little;
Servo thumb_a; // Thumb articulation servomotor

// JSON data buffer
StaticJsonDocument<250> jsonDocument;
char buffer[250];

// Function to send the HTML page to the client
void sendHtml() {
  String response = R"(
    <!DOCTYPE html>
<html>
<head>
  <title>ESP32 Bionic Hand Web Server</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: sans-serif;
      text-align: center;
      background-color: #f2f2f2;
      margin: 0;
      padding: 2em;
    }
    
    h1 {
      margin-bottom: 1.2em;
      color: #333;
      font-size: 2.5em;
    }

    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .card {
      background-color: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      padding: 2em;
      margin-bottom: 2em;
      text-align: center;
    }

    .card h2 {
      margin: 0;
      font-size: 1.8em;
      color: #333;
      margin-bottom: 1em;
    }
  </style>
</head>
      
<body>
  <div class="container">
    <h1>Bionic Hand Web Server</h1>

    <div class="card">
      <h2>Welcome to the Bionic Hand Web Server</h2>
      <p>Control your devices with ease and convenience.</p>
    </div>
  </div>
</body>
</html>
  )";
  server.send(200, "text/html", response);
}
// function to move the thumb servo motor
void moveThumb(int state){
  if(state == 1) { 
  // rotates from 0 degrees to 180 degrees
  // for (int pos = 0; pos <= 180; pos += 1) {
  //   // in steps of 1 degree
  //   thumb.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  thumb.write(0);
  delay(15);

  } else {
   // rotates from 180 degrees to 0 degrees
  // for (int pos = 180; pos >= 0; pos -= 1) {
  //   thumb.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  thumb.write(90);
  delay(15);
  }
}

// function to move the index servo motor
void moveIndex(int state){
  if(state == 1) { 
  // rotates from 0 degrees to 180 degrees
  // for (int pos = 0; pos <= 180; pos += 1) {
  //   // in steps of 1 degree
  //   index_f.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  index_f.write(0);
  delay(15);
  } else {
   // rotates from 180 degrees to 0 degrees
  // for (int pos = 180; pos >= 0; pos -= 1) {
  //   index_f.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  index_f.write(90);
  delay(15);
  }
}

// function to move the middle servo motor
void moveMiddle(int state){
  if(state == 1) { 
  // rotates from 0 degrees to 180 degrees
  // for (int pos = 0; pos <= 180; pos += 1) {
  //   // in steps of 1 degree
  //   middle.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  middle.write(0);
  delay(15);
  } else {
   // rotates from 180 degrees to 0 degrees
  // for (int pos = 180; pos >= 0; pos -= 1) {
  //   middle.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  middle.write(90);
  delay(15);
  }
}

// function to move the ring servo motor
void moveRing(int state){
  if(state == 1) { 
  // rotates from 0 degrees to 180 degrees
  // for (int pos = 0; pos <= 180; pos += 1) {
  //   // in steps of 1 degree
  //   ring.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  ring.write(0);
  delay(15);
  } else {
   // rotates from 180 degrees to 0 degrees
  // for (int pos = 180; pos >= 0; pos -= 1) {
  //   ring.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  ring.write(90);
  delay(15);
  }
}

// function to move the little servo motor
void moveLittle(int state){
  if(state == 1) { 
  // rotates from 0 degrees to 180 degrees
  // for (int pos = 0; pos <= 180; pos += 1) {
  //   // in steps of 1 degree
  //   little.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  little.write(0);
  delay(15);
  } else {
   // rotates from 180 degrees to 0 degrees
  // for (int pos = 180; pos >= 0; pos -= 1) {
  //   little.write(pos);
  //   delay(15); // waits 15ms to reach the position
  // }
  little.write(90);
  delay(15);
  }
}

void handlePost() {
  if (server.hasArg("plain") == false) {
    //handle error here
  }

  String body = server.arg("plain");
  Serial.println(body);
  deserializeJson(jsonDocument, body);
  
    // Get finger_control as an array with 5 elements
  int finger_control[5];
  JsonArray fingerArray = jsonDocument["control"];
  if (fingerArray.size() == 5) {
    for (int i = 0; i < 5; i++) {
      finger_control[i] = fingerArray[i].as<int>();
    }
  } else {
    // Handle error when the array size is not 5
  }
  Serial.print("Finger Contorl Array: ");
  for (int i = 0; i < 5; i++) {
  Serial.print(finger_control[i]);
  if (i < 4) {
    Serial.print(", ");
    }
  }
  Serial.println();

  // Process the finger_control array
  for (int i = 0; i < 5; i++) {
    // Access each element of the finger_control array
    int state = finger_control[i];
    // Call the respective motor control function based on the state (1 or 0)
    switch (i) {
      case 0:
        moveThumb(state);
        break;
      case 1:
        moveIndex(state);
        break;
      case 2:
        moveMiddle(state);
        break;
      case 3:
        moveRing(state);
        break;
      case 4:
        moveLittle(state);
        break;
      default:
        // Handle any additional cases or errors
        break;
    }
  }
  // Respond to the client
  server.send(200, "application/json", "{}");
}

void setup() {
  thumb.attach(THUMB_PIN);  // attaches the servo on ESP32 pin
  index_f.attach(INDEX_PIN);  // attaches the servo on ESP32 pin
  middle.attach(MIDDLE_PIN);  // attaches the servo on ESP32 pin
  ring.attach(RING_PIN);  // attaches the servo on ESP32 pin
  little.attach(LITTLE_PIN);  // attaches the servo on ESP32 pin
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, WIFI_CHANNEL);
  Serial.print("Connecting to WiFi ");
  Serial.print(WIFI_SSID);
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.println(" Connected!");

  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/", sendHtml);
  server.on("/api/control", HTTP_POST, handlePost);
  server.begin();
  Serial.println("HTTP server started (http://localhost:8180)");
}

void loop() {
  server.handleClient();
  delay(2);
  
}

void test(){
  moveThumb(1); // move the thumb servo motor
  delay(500); // waits 1 second
  moveThumb(0); // move the thumb servo motor
  delay(500); // waits 1 second

  moveIndex(1); // move the index servo motor
  delay(500); // waits 1 second
  moveIndex(0); // move the index servo motor
  delay(500); // waits 1 second

  moveMiddle(1); // move the middle servo motor
  delay(500); // waits 1 second
  moveMiddle(0); // move the middle servo motor
  delay(500); // waits 1 second

  moveRing(1); // move the ring servo motor
  delay(500); // waits 1 second
  moveRing(0); // move the ring servo motor
  delay(500); // waits 1 second

  moveLittle(1); // move the litle servo motor
  delay(500); // waits 1 second
  moveLittle(0); // move the litle servo motor
  delay(500); // waits 1 second
}
