# DEVELOPMENT PLAN

## A. Project Goal

The goal of GeoForge AI is to generate playable Geometry Dash levels from music using audio analysis and a machine learning model. The system will convert songs and user prompts into structured JSON that can be imported into the game for testing.

## B. Weekly Timeline (Class-by-Class Plan)
Day 1: Set up project folder, install Python, TensorFlow, and required libraries. Create basic file structure.

Day 2: Write code to load and process audio files. Extract simple features like tempo (BPM) and beat timing.

Day 3: Convert audio features into a structured format (arrays over time). Print and verify correctness.

Day 4: Design JSON format for level structure (positions, objects, timing). Create a simple manual JSON level.

Day 5: Build basic converter that turns simple beat data into level objects (no AI yet). Output JSON.

Day 6: Create Geode plugin or script that reads JSON and places objects into a level in Geometry Dash.

Day 7: Test full pipeline: audio → JSON → in-game level. Fix major bugs.

Day 8: Introduce TensorFlow model. Start with a simple model that maps beats → object types.

Day 9: Train/test model with small dataset (or simulated data). Integrate model output into JSON generator.

Day 10: Add prompt input (difficulty/style) and modify generation logic based on it.

Day 11: Improve level structure (spacing, patterns, difficulty scaling).

Day 12: Add export feature (convert existing levels to JSON for testing/training).

Day 13: Test multiple songs and prompts. Fix errors and improve consistency.

Day 14: Optional: add simple preview or visualization of generated level.

Day 15: Final testing, polish, and prepare presentation/demo.

C. Task List (Step-by-Step Build Plan)
Create project folder and organize files

Install and set up Python, TensorFlow, and audio libraries

Load audio files into program

Extract BPM and beat timing from audio

Convert audio data into structured arrays

Design JSON format for Geometry Dash levels

Write script to generate simple level JSON from beats

Build or configure Geode plugin for importing JSON

Test JSON import into Geometry Dash

Create basic TensorFlow model

Train model on sample or generated data

Connect model output to level generation system

Add prompt input for difficulty/style

Adjust generation logic based on prompt

Test full pipeline (audio → AI → JSON → game)

Debug and fix issues

Add optional features (preview, export, improvements)

D. Tools and Technologies
Programming Language: Python

Machine Learning: TensorFlow

Audio Processing: librosa (or similar)

Data Format: JSON

Game Integration: Geode (Geometry Dash modding framework)

Development Tools: VS Code or similar IDE

E. Risks, Challenges, and Backup Plan
Challenge 1: Audio analysis may be difficult or inaccurate

Solution: Start with simple BPM/beat detection instead of complex features

Challenge 2: Machine learning model may not produce good results

Solution: Use rule-based generation (beats → objects) as fallback

Challenge 3: JSON import into Geometry Dash may not work correctly

Solution: Focus on generating valid JSON and demonstrate output without full in-game integration

Backup Plan: If the AI model or game integration becomes too complex, the project will still function as a system that converts music into structured level JSON using rule-based logic instead of machine learning.