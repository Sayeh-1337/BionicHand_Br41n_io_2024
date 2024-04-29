
---

## ESP32 REST API Server for Bionic Hand Control using BCI EEG signals

This project showcases an ESP32 REST API server that allows you to control a bionic hand remotely. The server accepts control requests through a POST method with a JSON payload. Each element in the payload array represents the state of a finger on the bionic hand.

## Presentation Video
[Presentation Video](https://youtu.be/v1VJN46kQFo)

## Demo Video
Watch this YouTube video for an overview explanation of the project:

[Pantheon Demo](https://youtu.be/iF4NBgmN6jQ)

### Getting Started

To run this project, follow the steps below:

#### Prerequisites

- [PlatformIO](https://platformio.org) should be installed on your machine.
- [Wokwi for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode) is required for simulation.

#### Building

1. Install PlatformIO by following the [installation guide](https://docs.platformio.org/en/latest/core/installation/index.html).
2. Clone or download this project to your local machine.
3. Open the project in Visual Studio Code.
4. Run the following command in the terminal to build the project: `pio run`.

#### Simulating

1. Install the Wokwi for Visual Studio Code extension from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode).
2. Open the project directory in Visual Studio Code.
3. Press **F1** and select "Wokwi: Start Simulator" from the command palette.
4. Once the simulation is running, open http://localhost:8180 in your web browser to interact with the simulated HTTP server.

### Controlling the Bionic Hand

To control the bionic hand, send a POST request to the server with the following JSON payload:

```json
{
  "control": [1, 0, 0, 1, 0]
}
```

In the payload, each element in the `control` array corresponds to a finger on the hand. A value of `1` represents an flexed finger, while `0` represents a extended finger. Adjust the values in the array to control the desired finger positions.

### Simulation Script 
under the `dl_hackthon_2024` folder, you can find a `simulation.ipynb` python script that simulate EEG streaming then do predication task and sends a POST request to the server . You can run this script to control the bionic hand in the simulation.
for more interpretation of the AI model architecture and coding you can check the following [link](https://github.com/Ananas120/hackaton_2024) and reach @Ananas120 for more information is the expert.

```bash

### Conclusion

The ESP32 REST API server for bionic hand control offers a convenient way to remotely control the movements of a bionic hand. By sending a simple POST request with a JSON payload, you can adjust the finger positions according to your needs. This project can be further customized and integrated into larger systems for enhanced functionality and accessibility.

---