# Mapping Voice Commands to Robot Intents

## Introduction

In the previous chapter, we implemented speech recognition using OpenAI Whisper to convert voice commands into text. This chapter focuses on the critical next step: mapping recognized voice commands to specific robot intents and actions. This process involves understanding the user's request and translating it into actionable commands for the robot.

## Understanding Intent Mapping

### What is Intent Mapping

Intent mapping is the process of interpreting natural language voice commands and identifying the user's underlying intention. For example, when a user says "Go pick up the red ball," the intent is to navigate to an object and grasp it. The intent mapping system must understand:

- **Navigation intent**: Move toward a specific location/object
- **Grasping intent**: Pick up an object
- **Object identification**: Red ball (the target object)
- **Action sequence**: Navigate first, then grasp

### Intent Classification Approaches

There are several approaches to intent classification:

1. **Rule-based systems**: Predefined mappings from phrases to intents
2. **Machine learning classifiers**: Train models on labeled voice command datasets
3. **Large Language Model (LLM) reasoning**: Use LLMs for semantic understanding
4. **Hybrid approaches**: Combination of the above methods

## Rule-Based Intent Mapping

### Pattern Matching Approach

Rule-based systems use predefined patterns to match voice commands to intents:

```python
class RuleBasedIntentMapper:
    def __init__(self):
        self.intent_patterns = {
            "move_forward": [
                "go forward", "move ahead", "move forward", "go straight", 
                "move to the front", "proceed forward"
            ],
            "move_backward": [
                "go backward", "move back", "go back", "reverse", 
                "move to the rear", "go backwards"
            ],
            "turn_left": [
                "turn left", "rotate left", "go left", "pivot left", 
                "make a left turn", "turn to the left"
            ],
            "turn_right": [
                "turn right", "rotate right", "go right", "pivot right", 
                "make a right turn", "turn to the right"
            ],
            "pick_up": [
                "pick up", "grasp", "grab", "take", "lift", 
                "pick up the", "grasp the", "take the"
            ],
            "put_down": [
                "put down", "release", "place", "drop", "set down", 
                "put it down", "place it down"
            ],
            "navigate_to": [
                "go to", "navigate to", "move to", "walk to", 
                "go get", "bring me", "fetch", "go find"
            ],
            "stop": [
                "stop", "halt", "cease", "pause", "freeze", "wait"
            ]
        }
    
    def classify_intent(self, voice_command):
        """
        Classify intent using pattern matching
        """
        voice_command = voice_command.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in voice_command:
                    return {
                        "intent": intent,
                        "confidence": 0.8,  # High confidence for exact matches
                        "detected_phrase": pattern
                    }
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "detected_phrase": ""
        }
```

### Handling Synonyms and Variations

Expand the pattern matching to handle synonyms and linguistic variations:

```python
class EnhancedRuleBasedMapper(RuleBasedIntentMapper):
    def __init__(self):
        super().__init__()
        
        # Add variations and synonyms
        self.synonym_expansions = {
            "pick up": ["grasp", "grab", "take", "lift", "collect", "retrieve"],
            "put down": ["release", "place", "drop", "set down", "deposit", "position"],
            "go to": ["navigate to", "move to", "travel to", "head to", "move toward"],
            "turn": ["rotate", "pivot", "change direction", "spin"],
            "stop": ["halt", "pause", "wait", "freeze", "cease", "break"]
        }
    
    def expand_command(self, voice_command):
        """
        Expand command with known synonyms
        """
        expanded_commands = [voice_command]
        
        # Try different combinations of synonyms
        for phrase, synonyms in self.synonym_expansions.items():
            if phrase in voice_command:
                for synonym in synonyms:
                    expanded = voice_command.replace(phrase, synonym)
                    expanded_commands.append(expanded)
        
        return expanded_commands
    
    def classify_intent_enhanced(self, voice_command):
        """
        Classify intent using expanded pattern matching
        """
        # Get all expanded versions of the command
        expanded_commands = self.expand_command(voice_command)
        
        for command in expanded_commands:
            result = self.classify_intent(command)
            if result["intent"] != "unknown":
                # Boost confidence if expansion helped
                if command != voice_command:
                    result["confidence"] = min(result["confidence"] + 0.1, 1.0)
                return result
        
        # If none of the expansions worked, return original unknown
        return self.classify_intent(voice_command)
```

## Machine Learning-Based Intent Mapping

### Training Data Preparation

Create structured training data for intent classification:

```python
training_data = [
    {"text": "Go forward 1 meter", "intent": "move_forward"},
    {"text": "Move forward to the table", "intent": "move_forward"},
    {"text": "Turn left at the corner", "intent": "turn_left"},
    {"text": "Please turn to your left", "intent": "turn_left"},
    {"text": "Pick up the blue cup", "intent": "pick_up"},
    {"text": "Grab the red ball", "intent": "pick_up"},
    {"text": "Put down the object", "intent": "put_down"},
    {"text": "Release whatever you're holding", "intent": "put_down"},
    {"text": "Navigate to the kitchen", "intent": "navigate_to"},
    {"text": "Go to the red box", "intent": "navigate_to"},
    # Add many more examples for each intent class
]

def prepare_training_data(data):
    """
    Prepare training data for machine learning model
    """
    texts = [item["text"] for item in data]
    labels = [item["intent"] for item in data]
    
    return texts, labels
```

### Using Scikit-learn for Intent Classification

Implement a machine learning classifier for intent mapping:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class MLBasedIntentMapper:
    def __init__(self):
        self.pipeline = None
        self.intents = set()
    
    def train(self, training_data):
        """
        Train the ML model on provided data
        """
        texts, labels = prepare_training_data(training_data)
        self.intents = set(labels)
        
        # Create a pipeline with TF-IDF and Multinomial Naive Bayes
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # Use unigrams and bigrams
                stop_words='english',
                max_features=5000
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def classify_intent(self, voice_command):
        """
        Classify intent using the trained ML model
        """
        if not self.pipeline:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "probabilities": {}
            }
        
        # Predict the intent
        predicted_intent = self.pipeline.predict([voice_command])[0]
        
        # Get prediction probabilities
        probabilities = self.pipeline.predict_proba([voice_command])[0]
        classes = self.pipeline.classes_
        
        # Create probability dictionary
        prob_dict = dict(zip(classes, probabilities))
        
        # Get the confidence for the predicted intent
        confidence = prob_dict[predicted_intent]
        
        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "probabilities": prob_dict
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        """
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        """
        Load a pre-trained model from disk
        """
        self.pipeline = joblib.load(filepath)
```

## LLM-Based Intent Mapping

### Leveraging Large Language Models

Use LLMs for more nuanced intent classification with semantic understanding:

```python
import openai
from typing import Dict, List

class LLMIntentMapper:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.intents = [
            "move_forward", "move_backward", "turn_left", "turn_right",
            "pick_up", "put_down", "navigate_to", "stop", "unknown"
        ]
    
    def classify_intent(self, voice_command) -> Dict:
        """
        Use LLM to classify intent with natural language understanding
        """
        # Create a structured prompt for intent classification
        prompt = f"""
        You are a robot command interpreter. Your task is to identify the intended 
        action from a voice command. Choose the most appropriate intent from the 
        following list:

        Available intents:
        - move_forward: Robot should move in the forward direction
        - move_backward: Robot should move in the backward direction
        - turn_left: Robot should turn to its left
        - turn_right: Robot should turn to its right
        - pick_up: Robot should grasp/pick up an object
        - put_down: Robot should release/place down an object
        - navigate_to: Robot should navigate to a specified location or object
        - stop: Robot should cease current actions

        Voice command: "{voice_command}"

        Respond with only the intent name and a confidence score from 0 to 1. Format:
        Intent: [intent_name]
        Confidence: [confidence_score]
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Low temperature for consistent results
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            intent_line = [line for line in result.split('\n') if 'Intent:' in line]
            confidence_line = [line for line in result.split('\n') if 'Confidence:' in line]
            
            intent = intent_line[0].split(':')[1].strip() if intent_line else "unknown"
            confidence_str = confidence_line[0].split(':')[1].strip() if confidence_line else "0.0"
            
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.0
            
            # Validate the intent is in our defined list
            if intent not in self.intents:
                intent = "unknown"
                confidence = 0.0
            
            return {
                "intent": intent,
                "confidence": min(confidence, 1.0)  # Cap at 1.0
            }
            
        except Exception as e:
            print(f"Error with LLM intent classification: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0
            }
```

## Hybrid Intent Mapping System

### Combining Multiple Approaches

Create a hybrid system that combines rule-based, ML-based, and LLM-based approaches:

```python
class HybridIntentMapper:
    def __init__(self):
        self.rule_based = RuleBasedIntentMapper()
        self.ml_based = MLBasedIntentMapper()
        self.llm_based = LLMIntentMapper()
        
        # Thresholds for confidence
        self.confidence_thresholds = {
            "rule_based": 0.8,
            "ml_based": 0.7,
            "llm_based": 0.6
        }
    
    def classify_intent(self, voice_command):
        """
        Use a hybrid approach to classify intent
        """
        # 1. Try rule-based first (fast but may not catch all variations)
        rule_result = self.rule_based.classify_intent(voice_command)
        if rule_result["intent"] != "unknown" and rule_result["confidence"] >= self.confidence_thresholds["rule_based"]:
            return rule_result
        
        # 2. Try ML-based (good for variations)
        ml_result = self.ml_based.classify_intent(voice_command)
        if ml_result["intent"] != "unknown" and ml_result["confidence"] >= self.confidence_thresholds["ml_based"]:
            return ml_result
        
        # 3. Fall back to LLM-based (semantic understanding)
        llm_result = self.llm_based.classify_intent(voice_command)
        
        # If LLM result is strong enough, return it
        if llm_result["intent"] != "unknown" and llm_result["confidence"] >= self.confidence_thresholds["llm_based"]:
            return llm_result
        
        # If all approaches failed, return the best result among them
        results = [rule_result, ml_result, llm_result]
        best_result = max(results, key=lambda x: x["confidence"])
        
        return best_result
```

## Extracting Entities from Voice Commands

### Object and Attribute Recognition

Beyond intent classification, extract specific entities from the command:

```python
import re

class EntityExtractor:
    def __init__(self):
        # Define patterns for common entities
        self.color_patterns = [
            "red", "blue", "green", "yellow", "purple", "orange", 
            "black", "white", "gray", "brown", "pink"
        ]
        
        self.object_patterns = [
            "ball", "cup", "book", "box", "chair", "table", 
            "cube", "cylinder", "sphere", "robot", "object"
        ]
        
        self.location_patterns = [
            "kitchen", "bedroom", "living room", "office", 
            "table", "shelf", "desk", "corner"
        ]
    
    def extract_entities(self, voice_command):
        """
        Extract entities (objects, colors, locations) from the voice command
        """
        entities = {
            "colors": [],
            "objects": [],
            "locations": [],
            "numbers": [],
            "directions": []
        }
        
        # Extract colors
        for color in self.color_patterns:
            if color in voice_command.lower():
                entities["colors"].append(color)
        
        # Extract objects
        for obj in self.object_patterns:
            if obj in voice_command.lower():
                entities["objects"].append(obj)
        
        # Extract locations
        for loc in self.location_patterns:
            if loc in voice_command.lower():
                entities["locations"].append(loc)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', voice_command)
        entities["numbers"] = [int(n) for n in numbers]
        
        # Extract directions
        if "up" in voice_command.lower():
            entities["directions"].append("up")
        if "down" in voice_command.lower():
            entities["directions"].append("down")
        if "left" in voice_command.lower():
            entities["directions"].append("left")
        if "right" in voice_command.lower():
            entities["directions"].append("right")
        
        return entities

class EnhancedIntentMapper(HybridIntentMapper):
    def __init__(self):
        super().__init__()
        self.entity_extractor = EntityExtractor()
    
    def classify_with_entities(self, voice_command):
        """
        Classify intent and extract entities from the voice command
        """
        intent_result = self.classify_intent(voice_command)
        entities = self.entity_extractor.extract_entities(voice_command)
        
        return {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "entities": entities,
            "original_command": voice_command
        }
```

## Mapping Intents to Robot Actions

### Action Definition and Execution

Define how intents translate to specific robot actions:

```python
class RobotActionMapper:
    def __init__(self):
        self.action_templates = {
            "move_forward": {
                "ros_action": "nav2_msgs/NavigateToPose",
                "params": {"x_offset": 1.0, "y_offset": 0.0}
            },
            "move_backward": {
                "ros_action": "nav2_msgs/NavigateToPose", 
                "params": {"x_offset": -1.0, "y_offset": 0.0}
            },
            "turn_left": {
                "ros_action": "geometry_msgs/Twist",
                "params": {"angular_z": 0.5}
            },
            "turn_right": {
                "ros_action": "geometry_msgs/Twist",
                "params": {"angular_z": -0.5}
            },
            "pick_up": {
                "ros_action": "manipulation_msgs/PickUp",
                "params": {"object_name": "detected_object"}
            },
            "put_down": {
                "ros_action": "manipulation_msgs/PutDown",
                "params": {"object_name": "held_object"}
            },
            "navigate_to": {
                "ros_action": "nav2_msgs/NavigateToPose",
                "params": {"target_location": "specified_location"}
            },
            "stop": {
                "ros_action": "std_msgs/Empty",
                "params": {}
            }
        }
    
    def map_intent_to_action(self, intent_result, entities=None):
        """
        Map the classified intent to specific robot actions
        """
        intent = intent_result["intent"]
        entities = entities or intent_result.get("entities", {})
        
        if intent not in self.action_templates:
            return {
                "action_sequence": [],
                "error": f"Unknown intent: {intent}"
            }
        
        action_template = self.action_templates[intent]
        params = action_template["params"].copy()
        
        # Customize parameters based on extracted entities
        if intent == "pick_up" and entities and "objects" in entities:
            if entities["objects"]:
                params["object_name"] = entities["objects"][0]
        
        if intent == "navigate_to" and entities and "locations" in entities:
            if entities["locations"]:
                params["target_location"] = entities["locations"][0]
        
        # Build the complete action sequence
        action_sequence = [{
            "action_type": action_template["ros_action"],
            "parameters": params,
            "confidence": intent_result["confidence"]
        }]
        
        # Some intents might require multiple sequential actions
        if intent == "navigate_to" and "objects" in entities and entities["objects"]:
            # After navigating, might need to pick up the object
            action_sequence.append({
                "action_type": self.action_templates["pick_up"]["ros_action"],
                "parameters": {"object_name": entities["objects"][0]},
                "confidence": intent_result["confidence"] * 0.9  # Slightly lower confidence for chained actions
            })
        
        return {
            "action_sequence": action_sequence,
            "intent": intent,
            "confidence": intent_result["confidence"]
        }
```

## Quality Assurance and Validation

### Testing Intent Mapping

Create tests to validate the intent mapping system:

```python
class IntentMappingTester:
    def __init__(self, intent_mapper, action_mapper):
        self.intent_mapper = intent_mapper
        self.action_mapper = action_mapper
        
        # Define test cases
        self.test_cases = [
            {
                "command": "Go forward to the table",
                "expected_intent": "move_forward",
                "expected_entities": ["table"]
            },
            {
                "command": "Pick up the red ball",
                "expected_intent": "pick_up",
                "expected_entities": ["red", "ball"]
            },
            {
                "command": "Navigate to the kitchen and get the blue cup",
                "expected_intent": "navigate_to",
                "expected_entities": ["kitchen", "blue", "cup"]
            }
        ]
    
    def run_tests(self):
        """
        Run the test cases and validate results
        """
        results = []
        
        for test_case in self.test_cases:
            # Classify intent
            result = self.intent_mapper.classify_with_entities(test_case["command"])
            
            # Map to actions
            action_result = self.action_mapper.map_intent_to_action(
                result, result["entities"]
            )
            
            # Validate the result
            success = (
                result["intent"] == test_case["expected_intent"] and
                all(entity in result["entities"].get("objects", []) + 
                    result["entities"].get("colors", []) + 
                    result["entities"].get("locations", [])
                    for entity in test_case["expected_entities"])
            )
            
            results.append({
                "command": test_case["command"],
                "expected_intent": test_case["expected_intent"],
                "actual_intent": result["intent"],
                "expected_entities": test_case["expected_entities"],
                "actual_entities": result["entities"],
                "success": success,
                "confidence": result["confidence"],
                "actions": action_result
            })
        
        return results
    
    def generate_test_report(self, results):
        """
        Generate a report of test results
        """
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report = f"""
        Intent Mapping Test Report
        ==========================
        Total Tests: {total_tests}
        Passed: {passed_tests}
        Failed: {total_tests - passed_tests}
        Success Rate: {success_rate:.2%}
        
        Detailed Results:
        """
        
        for i, result in enumerate(results, 1):
            report += f"\n{i}. Command: '{result['command']}'\n"
            report += f"   Expected: {result['expected_intent']} | Actual: {result['actual_intent']}\n"
            report += f"   Success: {'✓' if result['success'] else '✗'} | Conf: {result['confidence']:.2f}\n"
        
        return report
```

## Troubleshooting Common Issues

### Ambiguous Commands

Handle commands that might map to multiple intents:

```python
def resolve_ambiguity(intent_result, entities, context=None):
    """
    Resolve ambiguity when multiple intents seem possible
    """
    if intent_result["confidence"] < 0.5:
        # Very low confidence suggests ambiguity
        # Use context or ask for clarification
        return request_clarification(intent_result, entities)
    
    return intent_result

def request_clarification(intent_result, entities):
    """
    Request clarification from user for ambiguous commands
    """
    return {
        "intent": "request_clarification",
        "confidence": 1.0,
        "message": "Could you clarify what you'd like me to do?",
        "options": [
            "Move forward?",
            "Turn in a direction?", 
            "Pick up an object?",
            "Go to a location?"
        ]
    }
```

## Summary

This chapter covered the essential process of mapping voice commands to robot intents. We explored multiple approaches to intent classification including rule-based, machine learning-based, and LLM-based methods, as well as hybrid systems that combine the strengths of each approach.

Key takeaways from this chapter:

- Intent mapping bridges the gap between natural language commands and robot actions
- Different approaches (rule-based, ML, LLM) have different strengths and weaknesses
- Entity extraction enhances intent understanding by identifying relevant objects, colors, and locations
- Quality assurance and testing ensure reliable intent classification
- Context and ambiguity resolution improve the robustness of the system

The next chapter will focus on implementing voice-robot interaction patterns, building on the foundation of speech recognition and intent mapping.