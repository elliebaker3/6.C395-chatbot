"""
Gradio Web Interface for MIT Course Catalog Chatbot

This script creates a web interface for your MIT course catalog chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    
    def chat(message, history):
        """
        TODO:Generate a response for the current message in a Gradio chat interface.
        
        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]
                           Example:
                           [
                               ["What schools offer Spanish?", "The Hernandez School..."],
                               ["Where is it located?", "The Hernandez School is in Roxbury..."]
                           ]

        Returns:
            str: The assistant's response to the current message.


        Note:
            - Gradio automatically:
                - Displays the user's message
                - Displays your returned response
                - Updates the chat history
                - Maintains the chat interface
            - You only need to:
                - Generate an appropriate response to the current message
                - Return that response as a string
        """
        # TODO: Generate and return response
        # Generate response using the chatbot
        response = chatbot.get_response(message, history)
        return response

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="MIT Course Catalog Assistant",
        description="Ask me about MIT courses! I can help you find courses based on prerequisites, schedules, distribution requirements (CI-H, HASS, REST), departments, instructors, and your interests. Since I am a free tier chatbot, I may give a 503 error when I'm busy. If that happens, please try again a few seconds later.",
        examples=[
            "I'm a 6-3 junior who needs a CI-H, prefers afternoon classes, and is interested in AI ethics",
            "What courses are available in the Computer Science department?",
            "Show me courses that satisfy HASS requirements"
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
