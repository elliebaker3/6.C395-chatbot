import json
from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN


class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        
        # Load MIT course catalog data
        try:
            with open('data.json', 'r') as f:
                self.course_data = json.load(f)
        except FileNotFoundError:
            self.course_data = {}
            print("Warning: data.json not found. MIT course catalog data will not be available.")
        
    def is_mit_course_question(self, user_input):
        """
        Classify whether the question is related to MIT course selection.
        
        Args:
            user_input (str): The user's question
            
        Returns:
            bool: True if the question is about MIT courses, False otherwise
        """
        # Classification prompt focused on MIT course selection
        classification_prompt = f"""Classify whether the following question is about MIT courses, course selection, the MIT course catalog, prerequisites, class schedules, distribution requirements (CI-H, HASS, REST), departments, instructors, or academic planning at MIT. 

Question: "{user_input}"

Respond with only "YES" if it's about MIT courses/course selection/catalog, or "NO" if it's about something else."""

        try:
            messages = [
                {"role": "system", "content": "You are a classification assistant. Respond with only YES or NO."},
                {"role": "user", "content": classification_prompt}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                result = response.choices[0].message.content.strip().upper()
            elif isinstance(response, dict) and 'choices' in response:
                result = response['choices'][0]['message']['content'].strip().upper()
            else:
                result = str(response).strip().upper()
            
            return "YES" in result
        except Exception:
            # Fallback: simple keyword check for MIT course-related terms
            mit_course_keywords = ['course', 'class', 'mit', 'catalog', 'prerequisite', 'schedule', 'department', 'instructor', 'ci-h', 'hass', 'rest', 'distribution', 'major', 'minor', 'enroll', 'registration', '6-', 'subject', 'units']
            return any(keyword.lower() in user_input.lower() for keyword in mit_course_keywords)
    
    def format_prompt(self, user_input, include_data=True):
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question
            include_data (bool): Whether to include school data in the context

        Returns:
            list: A list of messages in chat format
        
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"
        """
        # Create system message with MIT course catalog context
        system_message = """You are a helpful assistant specialized in helping MIT students navigate the MIT course catalog. 
You help students find courses that match their constraints and interests, including:
- Prerequisites
- Schedules and class times
- Departments and course numbers (e.g., 6-3, 18-06)
- Distribution requirements (CI-H, HASS, REST)
- Instructors
- Class formats (lecture, recitation, lab, etc.)
- Course descriptions and topics

When a student provides constraints (like "I'm a 6-3 junior who needs a CI-H, prefers afternoon classes, and is interested in AI ethics"), reason across all those dimensions to find matching courses.

Use the provided MIT course catalog data to answer questions accurately. If you don't know something from the data, say so. Be specific about course numbers, prerequisites, and requirements."""
        
        # Include MIT course catalog data in the context if available
        user_message = user_input
        if include_data and self.course_data:
            # Convert data to a readable string (limit size to avoid token limits)
            # Use compact JSON format to save tokens
            data_str = json.dumps(self.course_data, separators=(',', ':'))
            # Truncate if too long (keep first 3000 chars to stay within token limits)
            # This is a simple approach - could be optimized to extract relevant parts
            if len(data_str) > 3000:
                data_str = data_str[:3000] + "...\n[Note: MIT course catalog data truncated for length]"
            
            user_message = f"""MIT Course Catalog Data (from data.json):
{data_str}

User question: {user_input}"""
        
        # Format for Llama-3.1-Instruct chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return messages
        
    def get_response(self, user_input):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
        # First, classify if this is an MIT course question
        if not self.is_mit_course_question(user_input):
            return "I'm sorry, I can only help with questions about MIT courses, the course catalog, course selection, prerequisites, schedules, distribution requirements, and academic planning at MIT. Please ask me about MIT courses!"
        
        # Format the prompt with school data
        messages = self.format_prompt(user_input, include_data=True)
        
        # Generate response using the InferenceClient
        # The InferenceClient handles the chat template formatting automatically
        # Use chat_completion for conversational models
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            # Extract the response text - handle different response formats
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            elif isinstance(response, dict) and 'generated_text' in response:
                return response['generated_text']
            else:
                # If response format is unexpected, return string representation for debugging
                return str(response)
        except Exception as e:
            # If chat_completion fails, provide a helpful error message
            error_msg = str(e)
            return f"Error generating response: {error_msg}. Please check your HF_TOKEN and model access."
