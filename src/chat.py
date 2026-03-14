import json
from pathlib import Path
from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

from src import rag

DATA_PATH = "data.json"

def _data_path():
    """Path to data.json under project root."""
    root = Path(__file__).resolve().parent.parent
    return root / DATA_PATH

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
        
        # Load MIT course catalog data and ensure RAG index exists (from cache or build once)
        path = _data_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.course_data = json.load(f)
        except FileNotFoundError:
            self.course_data = {}
            print("Warning: data.json not found. MIT course catalog data will not be available.")
        if path.exists():
            rag.ensure_index(DATA_PATH)
        
    def is_mit_course_question(self, user_input, history):
        """
        Classify whether the question is related to MIT course selection.
        
        Args:
            user_input (str): The user's question
            
        Returns:
            bool: True if the question is about MIT courses, False otherwise
        """
        # iterate through the last 3 histories, if they exist and concatenate them into a single string
        print(f"Classifying question with {len(history) if history else 0} history items")  # ADD THIS

        history_str = ""
        if history:
            def _extract_text(history_items):
                texts = []
                for msg in history_items:
                    try:
                        # Handle Gradio tuple format: (user_msg, assistant_msg)
                        if isinstance(msg, tuple) and len(msg) == 2:
                            user_msg, assistant_msg = msg
                            user_text = user_msg if isinstance(user_msg, str) else str(user_msg)
                            assistant_text = assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg)
                            if user_text:
                                texts.append(f"User: {user_text}")
                            if assistant_text:
                                texts.append(f"Assistant: {assistant_text}")
                        # Handle dict format
                        elif isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, list):
                                # Handle list format: [{"type": "text", "text": "..."}, ...]
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        text = c.get("text", "")
                                        if text:
                                            texts.append(text)
                            elif isinstance(content, str):
                                # Handle string format: just use the string directly
                                if content:
                                    texts.append(content)
                    except Exception as e:
                        # Skip malformed messages
                        print(f"Warning: Could not extract text from history message: {e}")
                        continue
                return texts
            try:
                history_str = "\n".join(_extract_text(history[-3:]))
            except Exception as e:
                print(f"Warning: Could not process history: {e}")
                history_str = ""

        # Classification prompt focused on MIT course selection
        classification_prompt = f"""Classify whether the following question is about MIT courses, course selection, the MIT course catalog, prerequisites, class schedules, distribution requirements (CI-H, HASS, REST), departments, instructors, or academic planning at MIT. 
            Conversation history: {history_str}
            Question: "{user_input}"

            Respond with only "YES" if the user question is relevant in the conversation history
            or can be directly about MIT courses/course selection/catalog, or "NO" if it's about something else."""

        try:
            messages = [
                {"role": "system", "content": "You are a classification assistant. Respond with only YES or NO."},
                {"role": "user", "content": classification_prompt + "\n\n" + history_str}
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
            # Add these to the mit_course_keywords list
            mit_course_keywords = [
                'course', 'class', 'mit', 'catalog', 'prerequisite', 'schedule', 
                'department', 'instructor', 'ci-h', 'hass', 'rest', 'distribution', 
                'major', 'minor', 'enroll', 'registration', '6-', 'subject', 'units',
                'undergrad', 'graduate', 'yes', 'no', 'any', 'sure', 'ok', 'afternoon',
                'morning', 'evening', 'spring', 'fall', 'semester'
            ]
            return any(keyword.lower() in user_input.lower() for keyword in mit_course_keywords)
        
    
    def format_prompt(self, user_input, include_data=True, history=None):
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
        # Extract specific course numbers from user input and history
        import re
        course_numbers = set()
        
        # Pattern to match MIT course numbers (e.g., "6.4570", "18.06", "6.046J", "6-3")
        # Matches: digit(s) + dot/dash + digit(s) + optional letter
        course_pattern = r'\b(\d+[\.-]\d+[A-Z]?)\b'
        
        # Search in current user input
        matches = re.findall(course_pattern, user_input)
        course_numbers.update(matches)
        
        # Also search in conversation history for recently mentioned courses
        if history:
            for msg in history[-3:]:  # Check last 3 exchanges
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    text = (user_msg if isinstance(user_msg, str) else str(user_msg)) + " " + (assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg))
                    matches = re.findall(course_pattern, text)
                    course_numbers.update(matches)
        
        # Direct lookup: get full course data for any mentioned course numbers
        direct_course_chunks = []
        if include_data and self.course_data and course_numbers:
            classes = self.course_data.get("classes") or {}
            for course_num in course_numbers:
                # Try exact match first
                if course_num in classes:
                    course = classes[course_num]
                    # Format course as chunk (same format as RAG chunks)
                    parts = []
                    for key, value in course.items():
                        if isinstance(value, list):
                            value = ", ".join(map(str, value))
                        elif isinstance(value, dict):
                            value = json.dumps(value)
                        parts.append(f"{key}: {value}")
                    chunk_text = "\n".join(parts)
                    direct_course_chunks.append(chunk_text)
                else:
                    # Try matching with different formats (e.g., "6.4570" vs "6-4570")
                    normalized_num = course_num.replace(".", "-")
                    for course_id, course in classes.items():
                        if course_id.replace(".", "-") == normalized_num or course_id == normalized_num:
                            parts = []
                            for key, value in course.items():
                                if isinstance(value, list):
                                    value = ", ".join(map(str, value))
                                elif isinstance(value, dict):
                                    value = json.dumps(value)
                                parts.append(f"{key}: {value}")
                            chunk_text = "\n".join(parts)
                            direct_course_chunks.append(chunk_text)
                            break
        
        # RAG: retrieve relevant course chunks before building the system prompt
        retrieved_context = ""
        if include_data and self.course_data:
            retrieved_context = rag.get_relevant_context(user_input, top_k=8, data_path="data.json")
            if not retrieved_context.strip():
                retrieved_context = "(No relevant course excerpts retrieved from the catalog.)"
        
        # Combine direct lookups with RAG results, prioritizing direct lookups
        if direct_course_chunks:
            direct_context = "\n\n---\n\n".join(direct_course_chunks)
            if retrieved_context:
                # Prepend direct lookups (courses explicitly mentioned)
                retrieved_context = f"--- Courses explicitly mentioned in query (use these) ---\n\n{direct_context}\n\n--- Other relevant courses from catalog search ---\n\n{retrieved_context}"
            else:
                retrieved_context = f"--- Courses explicitly mentioned in query ---\n\n{direct_context}"

        # Load system prompt and insert retrieved chunks (step 5: build prompt)
        with open('prompts/system_prompt_draft.txt', 'r') as f:
            system_prompt_txt = f.read()
        
        # Add summarized history if it exists (from long conversations)
        if hasattr(self, '_history_summary') and self._history_summary:
            system_prompt_txt = (
                system_prompt_txt
                + "\n\n--- Summary of earlier conversation ---\n"
                + self._history_summary
            )
            # Clear it after using so it doesn't persist
            delattr(self, '_history_summary')
        
        if retrieved_context:
            system_prompt_txt = (
                system_prompt_txt
                + "\n\n--- Relevant course catalog excerpts (use only these when answering) ---\n\n"
                + retrieved_context
            )

        user_message = user_input
        messages = [
            {"role": "system", "content": system_prompt_txt}
        ]
        
        # Add conversation history to messages
        if history:
            for msg in history:
                try:
                    # Gradio ChatInterface passes history as list of tuples: [(user_msg, assistant_msg), ...]
                    if isinstance(msg, tuple) and len(msg) == 2:
                        user_msg, assistant_msg = msg
                        # Extract text from user message
                        if isinstance(user_msg, str):
                            user_text = user_msg
                        elif isinstance(user_msg, dict):
                            user_text = user_msg.get("content", "") or user_msg.get("text", "")
                        else:
                            user_text = str(user_msg) if user_msg else ""
                        
                        # Extract text from assistant message
                        if isinstance(assistant_msg, str):
                            assistant_text = assistant_msg
                        elif isinstance(assistant_msg, dict):
                            assistant_text = assistant_msg.get("content", "") or assistant_msg.get("text", "")
                        else:
                            assistant_text = str(assistant_msg) if assistant_msg else ""
                        
                        # Add both messages to history
                        if user_text:
                            messages.append({"role": "user", "content": user_text})
                        if assistant_text:
                            messages.append({"role": "assistant", "content": assistant_text})
                    # Handle dict format (if history was already converted)
                    elif isinstance(msg, dict):
                        role = msg.get("role")
                        content = msg.get("content")
                        if isinstance(content, list):
                            text = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
                        else:
                            text = content or ""
                        if role in ("system", "user", "assistant") and text:
                            messages.append({"role": role, "content": text})
                except Exception as e:
                    print(f"Warning: Could not process history message: {e}")
                    continue
        
        messages.append({"role": "user", "content": user_message})
        return messages

    def _summarize_history(self, history):
        """Convert Gradio history format to text for summarization."""
        with open('prompts/conversation_history_manager.txt', 'r') as f:
            conversation_history_manager_txt = f.read()
        try:
            # Convert history tuples to readable text
            history_text = ""
            for msg in history:
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    user_text = user_msg if isinstance(user_msg, str) else str(user_msg)
                    assistant_text = assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg)
                    history_text += f"User: {user_text}\nAssistant: {assistant_text}\n\n"
                else:
                    history_text += str(msg) + "\n"
            
            messages = [
                {"role": "system", "content": conversation_history_manager_txt},
                {"role": "user", "content": history_text}
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            else:
                return str(response)
        except Exception as e:
            print(f"Summarization error: {e}")
            return conversation_history_manager_txt

    def get_response(self, user_input, history=None):
        # print(f"\n--- New Message ---")
        # print(f"User input: {user_input}")  # ADD THIS
        # print(f"History length: {len(history) if history else 0}")  # ADD THIS
        # print(f"History contents: {history}")  # ADD THIS
        # First, classify if this is an MIT course question
        if not self.is_mit_course_question(user_input, history):
            print("Classified as: NOT an MIT course question")  # ADD THIS
            return "I'm sorry, I can only help with questions about MIT courses, the course catalog, course selection, prerequisites, schedules, distribution requirements, and academic planning at MIT. Please ask me about MIT courses!"
        print("Classified as: MIT course question ✓")  # ADD THIS
        if history and len(history) > 3:
            early_history = history[:-2]  # everything except last q&a
            recent_history = history[-2:]  # always keep the most recent q&a
            
            summarized_history = self._summarize_history(early_history)
            
            # Keep recent history in tuple format (Gradio format)
            # The summarized history will be added as context in format_prompt
            history = recent_history
            # Store summary separately to add in format_prompt if needed
            self._history_summary = summarized_history
            
            # print(f"SECOND: New history length: {len(history)}")
            # print(f"SECOND: New history: {history}")

        # Format the prompt with school data
        messages = self.format_prompt(user_input, include_data=True, history=history)
        
        # Generate response using the InferenceClient
        # The InferenceClient handles the chat template formatting automatically
        # Use chat_completion for conversational models
        try:
            print("Sending request to HuggingFace API...")  # ADD THIS
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
