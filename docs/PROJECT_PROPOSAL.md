# GeoForge AI

**Project Description**
GeoForge AI is a system that generates Geometry Dash–style levels from music using a combination of audio analysis and machine learning. The user provides a song and a short prompt describing the style or difficulty of the level, and the program produces a structured level design in JSON. This JSON is then imported directly into the game using a Geode plugin, allowing the generated level to be played and tested in real time. The project is designed for players and creators who want to quickly prototype rhythm-based levels without building everything manually.

**Problem & Purpose Statement**
Creating levels in rhythm-based platformers like Geometry Dash is time-consuming and requires careful synchronization with music. Many players want to experiment with level ideas or generate layouts quickly, but existing tools require manual placement of every object. This project solves that problem by automating the process, using audio features and user intent to generate level structures that can be directly tested in-game. It also demonstrates how structured data and machine learning can be combined to create interactive content.

**Main Features (MVP Features)**
The program allows the user to input a song and a prompt that influences the style of the generated level. It analyzes the audio to extract features such as beats, tempo, and intensity over time. A TensorFlow-based small model processes the audio features and prompt to generate a sequence of level actions or objects. These outputs are converted into a structured JSON format that represents the level layout. A Geode plugin reads this JSON and places the corresponding objects directly into a Geometry Dash level, enabling immediate testing. The system also includes the ability to export existing levels into JSON, allowing for debugging and the creation of training data.

**Stretch Features (Optional Extras)**
The system may include a simple visual preview of the generated level before it is imported into the game. It could also support saving and loading JSON level files, allowing users to revisit or modify generated levels. Additional enhancements may include more advanced prompt control, difficulty scaling, or support for multiple game modes within a single level. Another extension would be improving the model using exported real levels to refine generation quality.

**Program Flow Overview**
The user selects or provides a song and enters a prompt describing the desired level style. The program processes the audio and converts it into structured feature data over time. The TensorFlow model takes these features along with the prompt and generates a sequence of level actions. These actions are converted into a JSON representation of the level. The Geode plugin then imports this JSON into Geometry Dash, creating a playable level. The user can test the level in-game and optionally export it back into JSON for further refinement or training.

**Why I Chose This Project**
I chose this project because I am interested in both game development and AI systems, and I wanted to combine them into something interactive and meaningful. Geometry Dash level design is a process I am familiar with, and automating it presents both a technical challenge and a creative opportunity. This project allows me to explore machine learning, data structures, and real-time game integration in a way that results in a tangible and testable product.