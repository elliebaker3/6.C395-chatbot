import json
import re
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
        
        # Persistent context for specific professors/courses mentioned in conversation
        self._persistent_context = {
            "professors": {},  # {professor_name: [course_chunks]}
            "courses": {}     # {course_number: course_chunk}
        }

    def _is_generic_professor_label(self, value):
        """
        Filter out role labels that are not actual professor names.
        """
        text = (value or "").strip()
        if not text:
            return True

        normalized = re.sub(r"[^a-z\s]", "", text.lower()).strip()
        generic_patterns = [
            r"^professor$",
            r"^professors$",
            r"^prof$",
            r"^instructor$",
            r"^instructors$",
            r"^teacher$",
            r"^teachers$",
            r"^lecturer$",
            r"^lecturers$",
            r"^faculty$",
            r"^staff$",
            r"^teaching staff$",
            r"^instructional staff$",
            r"^course staff$",
            r"^the professor$",
            r"^the instructor$",
            r"^the lecturer$",
            r"^unknown instructor$",
            r"^unknown professor$",
            r"^professor name$",
            r"^instructor name$",
        ]
        return any(re.match(pattern, normalized) for pattern in generic_patterns)

    def _looks_like_professor_name(self, value):
        """
        Accept only strings that plausibly look like a person's name.
        """
        text = (value or "").strip()
        if not text:
            return False
        if self._is_generic_professor_label(text):
            return False
        if "?" in text or ":" in text or "\n" in text:
            return False

        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) > 60:
            return False

        # Allow names like "Sendhil Mullainathan", "S. Mullainathan", "J. Doyle"
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", normalized):
            return True
        if re.fullmatch(r"[A-Z]\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", normalized):
            return True
        if re.fullmatch(r"[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+", normalized):
            return True
        return False
        
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
    
    def _detect_specific_entities(self, user_input, history=None):
        """
        Detect if a specific professor or course number is mentioned.
        Returns tuple: (professor_names: set, course_numbers: set)
        """
        import re
        
        professor_names = set()
        course_numbers = set()
        
        # Pattern to match MIT course numbers
        course_pattern = r'\b(\d+[\.-]\d+[A-Z]?)\b'
        
        # Extract course numbers from current input
        matches = re.findall(course_pattern, user_input)
        course_numbers.update(matches)
        
        # Extract course numbers from history
        if history:
            for msg in history[-5:]:  # Check last 5 exchanges
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    text = (user_msg if isinstance(user_msg, str) else str(user_msg)) + " " + (assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg))
                    matches = re.findall(course_pattern, text)
                    course_numbers.update(matches)
        
        # Detect professor names - look for capitalized name patterns
        # Patterns that indicate a professor name
        professor_patterns = [
            r'(?:professor|prof|instructor|teaches?|teaching|taught by|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Full names
            r'(?:professor|prof|instructor|teaches?|teaching|taught by|by)\s+([A-Z][a-z]+)',  # Single names
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:teaches?|teaching|professor|prof)',  # Name before "teaches"
            r'what\s+(?:courses?|classes?)\s+(?:does|do|is|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:teach|teaching)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\'s\s+(?:courses?|classes?)',
            r'courses?\s+(?:by|from|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        
        # Also look for standalone capitalized names (2-3 words) that might be professors
        # This catches queries like "tell me what courses Sendhil Mullainathan is teaching"
        standalone_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        standalone_matches = re.findall(standalone_name_pattern, user_input)
        
        for match in standalone_matches:
            # Filter out common false positives
            false_positives = {'mit', 'fall', 'spring', 'summer', 'iap', 'course', 'class', 'student', 
                             'students', 'semester', 'term', 'year', 'department', 'major', 'minor'}
            if match.lower() not in false_positives and len(match.split()) >= 2:
                # Check if it appears in a context suggesting it's a professor
                context_before = user_input[:user_input.find(match)].lower()
                context_after = user_input[user_input.find(match) + len(match):].lower()
                if any(word in context_before or word in context_after for word in 
                       ['teach', 'teaching', 'professor', 'prof', 'instructor', 'course', 'class']) and not self._is_generic_professor_label(match):
                    professor_names.add(match.strip())
        
        for pattern in professor_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match:
                    # Filter false positives
                    if match.lower() not in ['mit', 'fall', 'spring', 'summer', 'iap'] and not self._is_generic_professor_label(match):
                        professor_names.add(match.strip())
        
        # Also check history for professor mentions
        if history:
            for msg in history[-5:]:
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    text = (user_msg if isinstance(user_msg, str) else str(user_msg)) + " " + (assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg))
                    for pattern in professor_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                match = match[0] if match else ""
                            if match and match.lower() not in ['mit', 'fall', 'spring', 'summer', 'iap'] and not self._is_generic_professor_label(match):
                                professor_names.add(match.strip())
        
        return professor_names, course_numbers

    def extract_entities_from_final_response(self, assistant_response):
        """
        Use an inference call on the FINAL assistant response text to extract
        structured entities for quick lookup.

        Returns:
            dict: {"courses": [...], "professors": [...]}
        """
        if not assistant_response or not str(assistant_response).strip():
            return {"courses": [], "professors": []}

        extractor_system = (
            "You extract MIT course numbers and professor names from assistant text. "
            "Return ONLY valid JSON with this exact schema: "
            "{\"courses\": [\"...\"], \"professors\": [\"...\"]}. "
            "Do not include explanations."
        )
        extractor_user = (
            "Extract the course numbers and professor names present in this text.\n\n"
            f"TEXT:\n{assistant_response}"
        )

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": extractor_system},
                    {"role": "user", "content": extractor_user},
                ],
                max_tokens=220,
                temperature=0.0,
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                raw = response.choices[0].message.content or ""
            elif isinstance(response, dict) and "choices" in response:
                raw = response["choices"][0]["message"]["content"] or ""
            else:
                raw = str(response)

            # Strip markdown fences if present.
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            # Keep only the first JSON object if extra text appears.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

            parsed = json.loads(raw)
            courses = parsed.get("courses", []) if isinstance(parsed, dict) else []
            professors = parsed.get("professors", []) if isinstance(parsed, dict) else []

            # Normalize and dedupe while preserving order.
            def _clean_list(items, entity_type=None):
                seen = set()
                cleaned = []
                for item in items or []:
                    text = str(item).strip()
                    if entity_type == "professor" and self._is_generic_professor_label(text):
                        continue
                    if text and text not in seen:
                        seen.add(text)
                        cleaned.append(text)
                return cleaned

            return {
                "courses": _clean_list(courses, entity_type="course"),
                "professors": _clean_list(professors, entity_type="professor"),
            }
        except Exception:
            # Fallback to deterministic extraction from final response only.
            profs, courses = self._detect_specific_entities(str(assistant_response), history=None)
            return {
                "courses": sorted(list(courses)),
                "professors": sorted(list(profs)),
            }

    def _parse_structured_model_output(self, raw_text):
        """
        Parse model output expected in the format:
        response: ...
        courses: ...
        professors: ...
        """
        text = (raw_text or "").strip()
        if not text:
            return {"response": "", "courses": [], "professors": [], "raw": raw_text}

        # Strip markdown fences if present.
        text = re.sub(r"^```(?:\w+)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # Try strict section parsing first.
        m = re.search(
            r"response:\s*(.*?)\n\s*courses:\s*(.*?)\n\s*professors:\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m:
            response_text = m.group(1).strip()
            courses_raw = m.group(2).strip()
            profs_raw = m.group(3).strip()
        else:
            # Tolerant fallback: parse whatever labels exist and NEVER leak metadata
            # into user-facing response.
            courses_match = re.search(r"(?im)^\s*courses:\s*(.*)$", text)
            profs_match = re.search(r"(?im)^\s*professors:\s*(.*)$", text)
            response_match = re.search(r"(?im)^\s*response:\s*(.*)$", text)

            courses_raw = courses_match.group(1).strip() if courses_match else ""
            profs_raw = profs_match.group(1).strip() if profs_match else ""

            # If "response:" label exists, prefer content after that label.
            if response_match:
                response_text = response_match.group(1).strip()
            else:
                # Otherwise, keep text before first metadata label.
                label_positions = []
                if courses_match:
                    label_positions.append(courses_match.start())
                if profs_match:
                    label_positions.append(profs_match.start())
                first_label_pos = min(label_positions) if label_positions else None
                if first_label_pos is not None:
                    response_text = text[:first_label_pos].strip()
                else:
                    response_text = text.strip()

            # If response still starts with "response:", strip it.
            response_text = re.sub(r"(?is)^\s*response:\s*", "", response_text).strip()

        def _is_none_like(value):
            text = (value or "").strip().lower()
            if not text:
                return True
            none_patterns = [
                r"^none$",
                r"^none listed$",
                r"^none mentioned$",
                r"^none provided$",
                r"^not mentioned$",
                r"^not listed$",
                r"^n/?a$",
                r"^null$",
                r"^\[\]$",
                r"^no courses?.*$",
                r"^no professors?.*$",
                r"^no instructors?.*$",
                r"^no names?.*$",
            ]
            return any(re.match(pattern, text) for pattern in none_patterns)

        def _split_entities(value, entity_type=None):
            if not value:
                return []
            if _is_none_like(value):
                return []
            items = [v.strip() for v in value.split(",")]
            seen = set()
            out = []
            for item in items:
                if not item or _is_none_like(item):
                    continue
                if entity_type == "professor" and not self._looks_like_professor_name(item):
                    continue
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out

        return {
            "response": response_text,
            "courses": _split_entities(courses_raw, entity_type="course"),
            "professors": _split_entities(profs_raw, entity_type="professor"),
            "raw": raw_text,
        }

    def get_response_bundle(self, user_input, history=None):
        """
        Generate the final assistant output in structured format and parse it.
        Returns:
            dict: {response: str, courses: list[str], professors: list[str], raw: str}
        """
        if not self.is_mit_course_question(user_input, history):
            msg = (
                "I'm sorry, I can only help with questions about MIT courses, the course catalog, "
                "course selection, prerequisites, schedules, distribution requirements, and academic "
                "planning at MIT. Please ask me about MIT courses!"
            )
            return {"response": msg, "courses": [], "professors": [], "raw": msg}

        if history and len(history) > 3:
            early_history = history[:-2]
            recent_history = history[-2:]
            summarized_history = self._summarize_history(early_history)
            history = recent_history
            self._history_summary = summarized_history

        messages = self.format_prompt(user_input, include_data=True, history=history)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Return your output in EXACTLY this format:\n"
                    "response: <the user-facing answer only>\n"
                    "courses: <comma-separated MIT course numbers mentioned anywhere in the response, or none>\n"
                    "professors: <comma-separated professor names mentioned anywhere in the response, or none>\n"
                    "You must ALWAYS include both the courses: line and the professors: line, even if the answer is none.\n"
                    "If the response mentions a course or professor, it must also appear in the corresponding line.\n"
                    "Do not add any extra sections."
                ),
            }
        )

        try:
            print("Sending request to HuggingFace API...")
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                raw_text = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                raw_text = response["choices"][0]["message"]["content"]
            elif isinstance(response, dict) and "generated_text" in response:
                raw_text = response["generated_text"]
            else:
                raw_text = str(response)

            parsed = self._parse_structured_model_output(raw_text)
            # Backfill structured metadata if the model omitted it.
            if not parsed.get("courses") or not parsed.get("professors"):
                extracted = self.extract_entities_from_final_response(parsed.get("response", ""))
                if not parsed.get("courses"):
                    parsed["courses"] = extracted.get("courses", [])
                if not parsed.get("professors"):
                    parsed["professors"] = extracted.get("professors", [])
            # Add truncation hint only to the user-facing response.
            response_text = parsed.get("response", "")
            if response_text and not response_text.strip().endswith((".", "!", "?", ":")):
                if any(indicator in response_text[-50:] for indicator in ["**", "Units:", "Schedule:", "Why it fits:"]):
                    parsed["response"] = response_text + "\n\n[Note: Response may have been truncated. Please ask for more details if needed.]"
            return parsed
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}. Please check your HF_TOKEN and model access."
            return {"response": error_msg, "courses": [], "professors": [], "raw": error_msg}

    def summarize_entity_relevance(self, entity_type, entity_name, conversation_text):
        """
        Generate a short (1-2 sentence or brief bullet) explanation of how an entity
        is relevant to the conversation so far.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You explain the relevance of an MIT course or professor to a student course-selection conversation. "
                        "Return 2-4 concise but informative bullet points OR 2-3 short sentences. "
                        "Mention why it mattered in the discussion, such as requirement fit, topic fit, instructor relevance, schedule fit, or prerequisite relevance. "
                        "Be specific, helpful, and grounded only in the provided conversation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Entity type: {entity_type}\n"
                        f"Entity name: {entity_name}\n\n"
                        f"Conversation so far:\n{conversation_text}\n\n"
                        "Output only the relevance summary."
                    ),
                },
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=140,
                temperature=0.3,
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                return (response.choices[0].message.content or "").strip()
            if isinstance(response, dict) and "choices" in response:
                return (response["choices"][0]["message"]["content"] or "").strip()
            return str(response).strip()
        except Exception:
            # Safe fallback if summarization call fails.
            if entity_type == "course":
                return f"- Mentioned as a potentially relevant course option in this discussion."
            return f"- Mentioned as a professor connected to courses discussed in this conversation."

    def dedupe_professor_name(self, candidate_name, existing_names):
        """
        Use the model to decide whether a new professor mention is the same person
        as one already present (e.g. 'Sendhil Mullainathan' vs 'S. Mullainathan').
        Returns the canonical existing name if matched, else the original candidate.
        """
        existing_names = [name for name in (existing_names or []) if name]
        candidate_name = (candidate_name or "").strip()
        if not candidate_name or self._is_generic_professor_label(candidate_name):
            return ""
        if not existing_names:
            return candidate_name

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You resolve whether professor names refer to the same person. "
                        "Return exactly one line. Either return one exact existing name from the provided list, "
                        "or return NEW if none match."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Candidate professor name: {candidate_name}\n"
                        f"Existing professor names: {', '.join(existing_names)}\n\n"
                        "If the candidate is the same professor as one of the existing names "
                        "(including abbreviations or initials), return that exact existing name. "
                        "Otherwise return NEW."
                    ),
                },
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=20,
                temperature=0.0,
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                answer = (response.choices[0].message.content or "").strip()
            elif isinstance(response, dict) and "choices" in response:
                answer = (response["choices"][0]["message"]["content"] or "").strip()
            else:
                answer = str(response).strip()

            answer = answer.strip()
            if answer == "NEW":
                return candidate_name
            for existing in existing_names:
                if answer == existing:
                    return existing
            return candidate_name
        except Exception:
            return candidate_name
    
    def _search_courses_by_professor(self, professor_name):
        """
        Search all courses for a specific professor name in inCharge field.
        Returns list of course chunks.
        """
        if not self.course_data:
            return []
        
        classes = self.course_data.get("classes") or {}
        matching_courses = []
        
        # Normalize professor name for matching (handle variations)
        prof_name_lower = professor_name.lower().strip()
        prof_parts = prof_name_lower.split()
        
        for course_id, course in classes.items():
            in_charge = course.get("inCharge", "")
            if not in_charge:
                continue
            
            # Check if professor name appears in inCharge field
            in_charge_lower = in_charge.lower()
            
            # Try different matching strategies
            # 1. Full name match (e.g., "sendhil mullainathan" in "Sendhil Mullainathan")
            if prof_name_lower in in_charge_lower:
                matching_courses.append((course_id, course))
                continue
            
            # 2. Last name match (common format: "J. Smith" or "Smith, J." or "Smith")
            if len(prof_parts) > 0:
                last_name = prof_parts[-1]
                # Check for last name with various formats
                # "J. Smith", "Smith, J.", "Smith", "Smith J.", etc.
                last_name_patterns = [
                    f". {last_name}",  # "J. Smith"
                    f"{last_name},",   # "Smith, J."
                    f"{last_name} ",   # "Smith " (followed by space)
                    f" {last_name}",  # " Smith" (preceded by space)
                    f"{last_name}.",  # "Smith."
                ]
                
                for pattern in last_name_patterns:
                    if pattern in in_charge_lower:
                        matching_courses.append((course_id, course))
                        break
                
                # Also try first initial + last name (e.g., "S. Mullainathan")
                if len(prof_parts) >= 2:
                    first_initial = prof_parts[0][0] if prof_parts[0] else ""
                    last_name = prof_parts[-1]
                    if first_initial and f"{first_initial}. {last_name}" in in_charge_lower:
                        matching_courses.append((course_id, course))
                        break
        
        # Remove duplicates
        seen = set()
        unique_courses = []
        for course_id, course in matching_courses:
            if course_id not in seen:
                seen.add(course_id)
                unique_courses.append((course_id, course))
        
        # Format matching courses as chunks
        course_chunks = []
        for course_id, course in unique_courses:
            parts = []
            for key, value in course.items():
                if isinstance(value, list):
                    value = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    value = json.dumps(value)
                parts.append(f"{key}: {value}")
            chunk_text = "\n".join(parts)
            course_chunks.append(chunk_text)
        
        return course_chunks
    
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
        # Detect specific professors and courses mentioned
        professor_names, course_numbers = self._detect_specific_entities(user_input, history)
        
        # Update persistent context with newly detected professors
        if professor_names and include_data and self.course_data:
            for prof_name in professor_names:
                if prof_name not in self._persistent_context["professors"]:
                    # Search for courses by this professor
                    prof_courses = self._search_courses_by_professor(prof_name)
                    if prof_courses:
                        self._persistent_context["professors"][prof_name] = prof_courses
                        print(f"Found {len(prof_courses)} courses for professor: {prof_name}")
        
        # Update persistent context with newly detected courses
        if course_numbers and include_data and self.course_data:
            classes = self.course_data.get("classes") or {}
            for course_num in course_numbers:
                if course_num not in self._persistent_context["courses"]:
                    # Try exact match first
                    if course_num in classes:
                        course = classes[course_num]
                        parts = []
                        for key, value in course.items():
                            if isinstance(value, list):
                                value = ", ".join(map(str, value))
                            elif isinstance(value, dict):
                                value = json.dumps(value)
                            parts.append(f"{key}: {value}")
                        chunk_text = "\n".join(parts)
                        self._persistent_context["courses"][course_num] = chunk_text
                    else:
                        # Try format variations
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
                                self._persistent_context["courses"][course_num] = chunk_text
                                break
        
        # Collect all persistent context (professors and courses from entire conversation)
        persistent_chunks = []
        
        # Add all courses from persistent context
        for course_num, course_chunk in self._persistent_context["courses"].items():
            persistent_chunks.append(course_chunk)
        
        # Add all courses from professors in persistent context
        for prof_name, prof_courses in self._persistent_context["professors"].items():
            persistent_chunks.extend(prof_courses)
        
        # Also get direct lookups for currently mentioned courses (in case not in persistent yet)
        direct_course_chunks = []
        if include_data and self.course_data and course_numbers:
            for course_num in course_numbers:
                if course_num in self._persistent_context["courses"]:
                    # Already in persistent context, skip
                    continue
                classes = self.course_data.get("classes") or {}
                # Try exact match first
                if course_num in classes:
                    course = classes[course_num]
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
                    # Try format variations
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
        
        # Combine persistent context, direct lookups, and RAG results
        all_direct_chunks = persistent_chunks + direct_course_chunks
        
        if all_direct_chunks:
            # Remove duplicates while preserving order
            seen = set()
            unique_chunks = []
            for chunk in all_direct_chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    unique_chunks.append(chunk)
            
            direct_context = "\n\n---\n\n".join(unique_chunks)
            if retrieved_context:
                # Prepend persistent/direct lookups (professors/courses explicitly mentioned)
                retrieved_context = f"--- Courses/Professors explicitly mentioned in conversation (ALWAYS use these) ---\n\n{direct_context}\n\n--- Other relevant courses from catalog search ---\n\n{retrieved_context}"
            else:
                retrieved_context = f"--- Courses/Professors explicitly mentioned in conversation (ALWAYS use these) ---\n\n{direct_context}"

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
                max_tokens=1024,  # Increased for history summarization
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
        bundle = self.get_response_bundle(user_input, history)
        return bundle.get("response", "")
