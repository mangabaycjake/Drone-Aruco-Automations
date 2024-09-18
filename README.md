# Drone-Aruco-Automations
*(Still be uploaded fully)*
These programs automates drone movement using python. It utilizes the [Drone-Aruco-Simulation](https://github.com/mangabaycjake/Drone-Aruco-Simulation) for executing the program. This has two parts:

## Traveller
This drone program memorizes a horizontal path along the map through visual data and navigate autonomously afterwards. 
It requires the presence on-scene of the ArUco markers to extract posiitonal data. 
It uses Artificial Neural Network (ANN) on training and decision making for movements on each path.  

## ArUco Follower
This drone program follows the Dummy from the simulation. The movements are ANN-based.
