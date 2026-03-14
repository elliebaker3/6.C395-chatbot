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
import re
from html import escape

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    
    classes = chatbot.course_data.get("classes") or {} if chatbot.course_data else {}

    def _extract_instructors(in_charge):
        if not in_charge:
            return []
        cleaned = str(in_charge)
        for season in ["Fall:", "Spring:", "Summer:", "IAP:"]:
            cleaned = cleaned.replace(season, "")
        return [token.strip() for token in cleaned.split(",") if token.strip()]

    def _format_summary_html(summary_text):
        text = (summary_text or "No relevance summary available yet.").strip()
        safe = escape(text)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        lines = [line.strip() for line in safe.splitlines() if line.strip()]

        bullet_lines = [line for line in lines if line.startswith("- ") or line.startswith("* ")]
        if bullet_lines and len(bullet_lines) == len(lines):
            items = "".join(f"<li>{line[2:].strip()}</li>" for line in bullet_lines)
            return f"<ul style='margin:0; padding-left:18px;'>{items}</ul>"

        return f"<div>{safe.replace(chr(10), '<br>')}</div>"
    
    def chat(message, history):
        """
        Generate a response for the current message in a Gradio chat interface.
        Also returns detected entities (courses/professors) for the right panel.
        """
        bundle = chatbot.get_response_bundle(message, history)
        response = bundle.get("response", "")
        course_numbers = bundle.get("courses", []) or []
        professor_names = bundle.get("professors", []) or []
        return response, course_numbers, professor_names
    
    def get_entity_info(entity_type, entity_name):
        """
        Retrieve information about a course or professor when button is clicked.
        """
        if entity_type == "course":
            course_data = None
            
            # Try exact match first
            if entity_name in classes:
                course_data = classes[entity_name]
            else:
                # Try format variations
                normalized_num = entity_name.replace(".", "-")
                for course_id, course in classes.items():
                    if course_id.replace(".", "-") == normalized_num or course_id == normalized_num:
                        course_data = course
                        break
            
            if course_data:
                # Pretty, consistent formatting from data.json
                level = "Undergraduate" if course_data.get("level") == "U" else "Graduate" if course_data.get("level") == "G" else "N/A"
                units = f"{course_data.get('lectureUnits', 0)}-{course_data.get('labUnits', 0)}-{course_data.get('preparationUnits', 0)}"
                info = f"### {entity_name} — {course_data.get('name', 'N/A')}\n\n"
                info += f"- **Level:** {level}\n"
                info += f"- **Units:** {units}\n"
                info += f"- **Terms:** {course_data.get('terms', 'N/A')}\n"
                info += f"- **Prerequisites:** {course_data.get('prereqs', 'None')}\n"
                info += f"- **HASS / Comms / GIR:** {course_data.get('hass', 'None')} / {course_data.get('comms', 'None')} / {course_data.get('gir', 'None')}\n"
                info += f"- **Instructor(s):** {course_data.get('inCharge', 'N/A')}\n"
                if course_data.get("rating"):
                    info += f"- **Rating:** {course_data.get('rating')}/7"
                    if course_data.get("hours"):
                        info += f" ({course_data.get('hours')} hrs/week)"
                    info += "\n"
                info += f"\n**Description**\n\n{course_data.get('description', 'N/A')}\n"
                return info
            else:
                return f"Course {entity_name} not found in the catalog."
        elif entity_type == "professor":
            prof_lower = entity_name.lower().strip()
            matches = []
            for course_id, course in classes.items():
                in_charge = str(course.get("inCharge", ""))
                if not in_charge:
                    continue
                if prof_lower in in_charge.lower():
                    matches.append((course_id, course))
                    continue
                for instructor in _extract_instructors(in_charge):
                    if prof_lower == instructor.lower():
                        matches.append((course_id, course))
                        break

            if not matches:
                return f"No courses found for professor {entity_name}."

            info = f"### Professor: {entity_name}\n\n"
            info += f"Found **{len(matches)}** matching course(s).\n\n"
            for _, course in sorted(matches, key=lambda x: str(x[1].get("number", x[0]))):
                num = course.get("number", "N/A")
                name = course.get("name", "N/A")
                level = "Undergraduate" if course.get("level") == "U" else "Graduate" if course.get("level") == "G" else "N/A"
                terms = course.get("terms", "N/A")
                units = f"{course.get('lectureUnits', 0)}-{course.get('labUnits', 0)}-{course.get('preparationUnits', 0)}"
                info += f"- **{num}** — {name} ({level}, {terms}, {units} units)\n"
            return info
        return "Invalid entity type."

    
    
    # Custom CSS for professional styling
    custom_css = """
    /* Main container styling - full screen grey gradient background */
    .gradio-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        min-height: 100vh !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow-y: auto !important;
    }
    
    /* Body and html to ensure scrolling works */
    body, html {
        margin: 0 !important;
        padding: 0 !important;
        overflow-y: auto !important;
        scroll-behavior: smooth !important;
    }
    
    /* Ensure page loads at top */
    body {
        scroll-top: 0 !important;
    }
    
    /* Description text - bigger and more prominent */
    .description, .description p, .description-text, .markdown p {
        font-size: 18px !important;
        line-height: 1.7 !important;
        color: #2c3e50 !important;
        font-weight: 400 !important;
        padding: 20px 0 !important;
        margin-bottom: 10px !important;
    }
    
    /* Title styling - more impressive */
    h1, .h1, .markdown h1 {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        margin-bottom: 20px !important;
        letter-spacing: -0.8px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Chat lane should use full width without horizontal scrolling */
    .chatbot, .chatbot-container, [class*="chatbot"], [class*="message-container"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Remove extra visual wrapper boxes around messages */
    .message-wrap,
    [class*="message-wrap"],
    [class*="message-row"],
    [class*="bubble-wrap"],
    .chatbot > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 10px 0 !important;
        max-width: 100% !important;
    }

    /* Actual message bubble styling */
    .message {
        border-radius: 16px !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        max-width: 100% !important;
        width: 100% !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
        background: transparent !important;
        border: none !important;
    }
    
    .message.user {
        display: flex !important;
        justify-content: flex-end !important;
        margin-left: auto !important;
    }
    .message.assistant {
        display: flex !important;
        justify-content: flex-start !important;
        margin-right: auto !important;
    }

    /* Keep outer rows invisible and use them only for alignment */
    .user-message, .assistant-message,
    .message.user, .message.assistant,
    .message.user > div, .message.assistant > div,
    [class*="message-row"], [class*="message-wrap"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .message.user [class*="message-wrap"],
    .message.assistant [class*="message-wrap"],
    .message.user .message-wrap,
    .message.assistant .message-wrap {
        width: 100% !important;
        display: flex !important;
    }

    .message.user [class*="message-wrap"],
    .message.user .message-wrap {
        justify-content: flex-end !important;
    }

    .message.assistant [class*="message-wrap"],
    .message.assistant .message-wrap {
        justify-content: flex-start !important;
    }

    /* Style only the actual visible bubble */
    .message.user .message-content,
    .message.user [class*="message-content"],
    .message.assistant .message-content,
    .message.assistant [class*="message-content"] {
        display: inline-block !important;
        width: auto !important;
        max-width: 44% !important;
        padding: 13px 16px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
    }

    .message.user .message-content,
    .message.user [class*="message-content"] {
        background: linear-gradient(135deg, #6d5efc 0%, #5a4be7 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .message.assistant .message-content,
    .message.assistant [class*="message-content"] {
        background: #eef1f5 !important;
        color: #1f2937 !important;
        border: none !important;
    }

    .message p, .message div, .message span {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
    }

    
    /* Input area styling - more prominent */
    textarea, input[type="text"], .input-text {
        border-radius: 12px !important;
        border: 2px solid #e1e8ed !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        background: #ffffff !important;
    }
    
    textarea:focus, input[type="text"]:focus, .input-text:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Button styling - more professional */
    button, .btn {
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        font-size: 15px !important;
    }
    
    button:hover, .btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Primary button - gradient */
    button.primary, .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }

    /* Subtle "How it works" control */
    #how-it-works-btn, #how-it-works-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #6b7280 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 2px 0 !important;
        min-height: auto !important;
    }
    #how-it-works-btn:hover, #how-it-works-btn button:hover {
        transform: none !important;
        box-shadow: none !important;
        color: #4b5563 !important;
        text-decoration: underline;
    }

    /* Quick lookup expandable cards */
    .entity-details {
        margin: 8px 0 !important;
    }
    .entity-summary-btn {
        list-style: none !important;
        cursor: pointer !important;
        display: inline-block !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin: 0 6px 6px 0 !important;
        user-select: none !important;
        border: none !important;
    }
    .entity-summary-btn::-webkit-details-marker {
        display: none !important;
    }
    .entity-summary-btn-course {
        background: #667eea !important;
    }
    .entity-summary-btn-prof {
        background: #764ba2 !important;
    }
    .entity-detail-card {
        margin-top: 8px !important;
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
        color: #374151 !important;
        line-height: 1.55 !important;
    }
    
    /* Example buttons - more polished and visible */
    .example, .example-btn, button[class*="example"], 
    [class*="example-button"], [class*="example-btn"] {
        background: #ffffff !important;
        border: 2px solid #e1e8ed !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        margin: 8px 6px !important;
        transition: all 0.2s ease !important;
        color: #2c3e50 !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        display: inline-block !important;
        cursor: pointer !important;
    }
    
    .example:hover, .example-btn:hover, 
    button[class*="example"]:hover, [class*="example-button"]:hover {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Ensure example section is visible */
    [class*="examples"], [class*="example-container"],
    [class*="example-row"], [class*="example-list"] {
        display: block !important;
        visibility: visible !important;
        margin: 20px 0 !important;
    }
    
    /* Make retry/undo/clear buttons less prominent */
    button[class*="retry"], button[class*="undo"], button[class*="clear"],
    [class*="retry-btn"], [class*="undo-btn"], [class*="clear-btn"],
    button[aria-label*="retry"], button[aria-label*="undo"], button[aria-label*="clear"] {
        opacity: 0.5 !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
        background: #f7f8fa !important;
        border: 1px solid #e1e8ed !important;
        color: #6c757d !important;
        margin: 4px !important;
    }
    
    button[class*="retry"]:hover, button[class*="undo"]:hover, button[class*="clear"]:hover {
        opacity: 0.7 !important;
        background: #e8ecf1 !important;
    }
    
    /* Ensure examples are visible and clickable */
    [class*="example"], [class*="examples"],
    [data-testid*="example"], [class*="example-row"],
    [class*="example-list"], [class*="example-container"],
    button[class*="example"], a[class*="example"],
    div[class*="example"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin: 10px 0 !important;
    }
    
    /* Style example buttons to be more visible */
    button.example-btn, button[class*="example-btn"] {
        background: #667eea !important;
        color: white !important;
        border: none !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        margin: 8px 8px 8px 0 !important;
        transition: all 0.2s !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: left !important;
        line-height: 1.4 !important;
    }
    
    button.example-btn:hover, button[class*="example-btn"]:hover {
        background: #5568d3 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Overall spacing and layout */
    .main {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 40px 30px !important;
    }
    
    /* Ensure main content area takes full width */
    .gradio-container > div,
    .gradio-container > .container,
    [class*="container"]:not([class*="chatbot"]),
    .gradio-container .wrap,
    .gradio-container .wrap-inner,
    .gradio-container [class*="wrap"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Ensure Row and Column components take full width */
    [class*="row"], [class*="Row"],
    [class*="column"], [class*="Column"],
    div[class*="row"], div[class*="Row"],
    div[class*="column"], div[class*="Column"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Override any Gradio default max-width constraints */
    .gradio-container,
    .gradio-container * {
        box-sizing: border-box !important;
    }
    
    /* Ensure the Blocks container itself is full width */
    .gradio-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Chat interface - no fixed vertical sizing */
    .chat-container, .chat-interface, [class*="chat"] {
        min-height: 0 !important;
        height: auto !important;
    }
    
    /* Chatbot area - content-driven height */
    .chatbot, .chatbot-container {
        min-height: 0 !important;
        height: auto !important;
    }
    
    /* Hide Gradio footer elements completely */
    footer, .footer, [class*="footer"], 
    [class*="api"], [id*="api"],
    [class*="gradio-link"], a[href*="gradio"],
    .gradio-link, #gradio-link,
    .gradio-container footer,
    .gradio-container .footer,
    .gradio-container a[href*="api"],
    .gradio-container a[href*="gradio"],
    div[class*="footer"],
    div[id*="footer"],
    span[class*="gradio"],
    a[target="_blank"][href*="gradio"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure chat area takes up more space */
    .chatbot, .chatbot-container, [class*="chatbot"] {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Chat messages container - content-driven */
    [class*="chat-message"], [class*="message-container"] {
        min-height: 0 !important;
        max-height: none !important;
    }
    
    /* Hide the initial empty chat placeholder/shadowy box */
    .chatbot-empty, [class*="empty"], [class*="placeholder"],
    .chatbot:empty::before, [class*="chat"]:empty::before,
    div[class*="chat"]:empty, div[class*="message"]:empty,
    [class*="empty-state"], [class*="no-messages"],
    .chatbot-container:empty, .chat-interface:empty {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide the initial chat box shadow/placeholder that appears before first message */
    .chatbot > div:empty,
    .chatbot > div:only-child:empty,
    [class*="chat"] > div:empty:first-child,
    [class*="chatbot"] [class*="message"]:empty,
    [class*="chatbot"] [class*="bubble"]:empty {
        display: none !important;
        opacity: 0 !important;
    }
    
    /* Ensure chat area shows properly even when empty */
    .chatbot, [class*="chatbot"] {
        background: transparent !important;
    }
    
    /* Card-like containers */
    .panel, .card {
        background: #ffffff !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e1e8ed !important;
    }
    
    /* Scrollbar styling - more polished */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #653a8f 100%);
    }
    """
    
    # Create Gradio interface with custom styling
    with gr.Blocks(
        css=custom_css, 
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="slate",
            font=("ui-sans-serif", "system-ui", "sans-serif"),
        )
    ) as demo:
        gr.Markdown("""
        # MIT Course Catalog Assistant
        
        <div style="font-size: 18px; line-height: 1.7; color: #2c3e50; padding: 20px 0; margin-bottom: 10px;">
        <p style="margin-bottom: 16px;">
        Ask me about MIT courses! I can help you find courses based on prerequisites, schedules, distribution requirements (CI-H, HASS, REST), departments, instructors, and your interests. Whether you're looking for courses that match your major requirements, fit your schedule, or align with your academic goals, I'm here to help navigate MIT's extensive course catalog.
        </p>
        
        <p style="font-size: 14px; color: #718096; font-style: italic; margin-bottom: 20px;">
        Note: Since I am a free tier chatbot, I may give a 503 error when I'm busy. If that happens, please try again a few seconds later.
        </p>
        </div>
        """)

        how_it_works_state = gr.State(False)
        how_it_works_button = gr.Button("How it works ▼", variant="secondary", size="sm", elem_id="how-it-works-btn")
        how_it_works_text = gr.Markdown(
            """
            <div style="margin-top: 10px; padding: 12px 0; color: #4a5568; line-height: 1.7; font-size: 16px;">
            This assistant combines an advanced <strong>RAG (Retrieval-Augmented Generation)</strong> pipeline with semantic embeddings and FAISS vector search to instantly scan thousands of MIT catalog entries by meaning, not just keywords. A classifier first routes course-related queries, the retrieval layer surfaces the most relevant course context, and a large language model synthesizes that evidence with conversation memory to deliver precise, context-aware recommendations across prerequisites, schedules, requirements, and instructor preferences.
            </div>
            """,
            visible=False
        )

        def toggle_how_it_works(is_open):
            new_state = not is_open
            button_label = "How it works ▲" if new_state else "How it works ▼"
            return new_state, gr.update(value=button_label), gr.update(visible=new_state)

        how_it_works_button.click(
            toggle_how_it_works,
            inputs=[how_it_works_state],
            outputs=[how_it_works_state, how_it_works_button, how_it_works_text]
        )
        
        # Create two-column layout
        with gr.Row():
            # Left column: Description, Chatbot, Examples
            with gr.Column(scale=2):
                example_questions = [
                    "I'm a 6-3 junior who needs a CI-H, prefers afternoon classes, and is interested in AI ethics",
                    "What courses are available in the Computer Science department?",
                    "Show me courses that satisfy HASS requirements"
                ]
                
                # Chat interface - wrap to capture entities
                chat_starter_text = gr.HTML(
                    "<div id='chat-starter-text' style='color:#9ca3af; font-style: italic; margin: 0 0 10px 0;'>Send a message below to start interacting.</div>"
                )
                chatbot_ui = gr.Chatbot(value=[], bubble_full_width=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a message...",
                        show_label=False,
                        container=False,
                        elem_id="chat-input-textbox",
                        scale=8
                    )
                    enter_btn = gr.Button("Enter", variant="primary", scale=1, elem_id="chat-enter-btn")
                
                # Store all detected entities
                all_courses_state = gr.State([])
                all_professors_state = gr.State([])
                entity_summaries_state = gr.State({})
                pending_message_state = gr.State("")
                
                def add_user_message(message, history):
                    history = history or []
                    if not message or not message.strip():
                        return history, gr.update(value=message or "", interactive=True), gr.update(interactive=True), ""
                    new_history = history + [(message, None)]
                    return new_history, gr.update(value="", interactive=False), gr.update(interactive=False), message

                # Process chat and update entities after typing placeholder is shown
                def generate_bot_response(history, pending_message, existing_courses, existing_professors, existing_summaries):
                    history = history or []
                    pending_message = pending_message or ""
                    existing_courses = existing_courses or []
                    existing_professors = existing_professors or []
                    existing_summaries = existing_summaries or {}
                    if not history or not pending_message:
                        return history, existing_courses, existing_professors, existing_summaries, gr.update(interactive=True, value=""), gr.update(interactive=True), ""

                    prior_history = history[:-1]
                    response, courses, professors = chat(pending_message, prior_history)
                    new_history = prior_history + [(pending_message, response)]

                    # Merge entities and preserve across whole conversation.
                    updated_courses = sorted(list(set(existing_courses + (courses or []))))
                    updated_professors = list(existing_professors)
                    for prof in (professors or []):
                        canonical = chatbot.dedupe_professor_name(prof, updated_professors)
                        if canonical not in updated_professors:
                            updated_professors.append(canonical)
                    updated_professors = sorted(set(updated_professors))

                    # Summaries are generated only for newly populated entities.
                    updated_summaries = dict(existing_summaries)
                    history_text_lines = []
                    for item in new_history:
                        if isinstance(item, tuple) and len(item) == 2:
                            u, a = item
                            history_text_lines.append(f"User: {u}")
                            history_text_lines.append(f"Assistant: {a}")
                    conversation_text = "\n".join(history_text_lines)

                    new_courses = [c for c in (courses or []) if c not in existing_courses]
                    for course in new_courses:
                        key = f"course::{course}"
                        if key not in updated_summaries:
                            updated_summaries[key] = chatbot.summarize_entity_relevance("course", course, conversation_text)

                    new_professors = []
                    for p in (professors or []):
                        canonical = chatbot.dedupe_professor_name(p, existing_professors + new_professors)
                        if canonical not in existing_professors and canonical not in new_professors:
                            new_professors.append(canonical)
                    for prof in new_professors:
                        key = f"professor::{prof}"
                        if key not in updated_summaries:
                            updated_summaries[key] = chatbot.summarize_entity_relevance("professor", prof, conversation_text)
                    return new_history, updated_courses, updated_professors, updated_summaries, gr.update(interactive=True, value=""), gr.update(interactive=True), ""
                
                msg_input.submit(
                    add_user_message,
                    [msg_input, chatbot_ui],
                    [chatbot_ui, msg_input, enter_btn, pending_message_state],
                    show_progress="hidden"
                ).then(
                    generate_bot_response,
                    [chatbot_ui, pending_message_state, all_courses_state, all_professors_state, entity_summaries_state],
                    [chatbot_ui, all_courses_state, all_professors_state, entity_summaries_state, msg_input, enter_btn, pending_message_state],
                    show_progress="hidden"
                )

                enter_btn.click(
                    add_user_message,
                    [msg_input, chatbot_ui],
                    [chatbot_ui, msg_input, enter_btn, pending_message_state],
                    show_progress="hidden"
                ).then(
                    generate_bot_response,
                    [chatbot_ui, pending_message_state, all_courses_state, all_professors_state, entity_summaries_state],
                    [chatbot_ui, all_courses_state, all_professors_state, entity_summaries_state, msg_input, enter_btn, pending_message_state],
                    show_progress="hidden"
                )
                
                # Example questions section
                gr.Markdown("""
                <div style="font-size: 16px; color: #4a5568; margin: 20px 0 10px 0;">
                <strong>Try these example questions:</strong> Click any example below to get started.
                </div>
                """)
                
                gr.Examples(
                    examples=example_questions,
                    inputs=[msg_input],
                    label=""
                )
                
                # Add JavaScript to ensure page loads at top and allows scrolling
                gr.HTML("""
                <script>
                    // Scroll to top on page load
                    window.addEventListener('load', function() {
                        window.scrollTo(0, 0);
                        document.body.scrollTop = 0;
                        document.documentElement.scrollTop = 0;
                    });
                    // Also scroll to top immediately if DOM is ready
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', function() {
                            window.scrollTo(0, 0);
                        });
                    } else {
                        window.scrollTo(0, 0);
                    }
                </script>
                """)
            
            # Right column: Entity buttons and info display
            with gr.Column(scale=1):
                gr.Markdown("### Course & Professor Quick Lookup")
                gr.Markdown("Click on any course number or professor name mentioned in the conversation to see details and why it was relevant.")
                
                # Entity buttons container
                entity_buttons_html = gr.HTML(
                    "<div style='margin: 10px 0; color: #9ca3af; font-style: italic;'>Start interacting to see this populate.</div>"
                )

                # Update buttons HTML and expandable cards for detected entities
                def update_buttons_with_handlers(courses, professors, summaries):
                    # Filter to only show detected entities (non-empty)
                    courses = [c for c in (courses or []) if c and c.strip()]
                    professors = [p for p in (professors or []) if p and p.strip()]
                    summaries = summaries or {}
                    
                    if not courses and not professors:
                        return "<div style='margin: 10px 0; color: #9ca3af; font-style: italic;'>Start interacting to see this populate.</div>"
                    
                    html = '<div class="entity-buttons-group" style="margin: 10px 0;">'
                    if courses:
                        html += '<div style="margin-bottom: 15px;"><strong>Courses:</strong><br>'
                        for idx, course in enumerate(sorted(set(courses))):  # Remove duplicates and sort
                            summary = summaries.get(f"course::{course}", "")
                            summary_html = _format_summary_html(summary)
                            detail_id = f"course-detail-{idx}"
                            html += (
                                "<details class='entity-details'>"
                                f"<summary class='entity-summary-btn entity-summary-btn-course'>{escape(course)}</summary>"
                                f"<div id='{detail_id}' class='entity-detail-card'><div style='font-weight:700; margin-bottom:6px;'>Course: {escape(course)}</div>{summary_html}</div>"
                                "</details>"
                            )
                        html += '</div>'
                    if professors:
                        html += '<div><strong>Professors:</strong><br>'
                        for idx, prof in enumerate(sorted(set(professors))):  # Remove duplicates and sort
                            summary = summaries.get(f"professor::{prof}", "")
                            summary_html = _format_summary_html(summary)
                            detail_id = f"prof-detail-{idx}"
                            html += (
                                "<details class='entity-details'>"
                                f"<summary class='entity-summary-btn entity-summary-btn-prof'>{escape(prof)}</summary>"
                                f"<div id='{detail_id}' class='entity-detail-card'><div style='font-weight:700; margin-bottom:6px;'>Professor: {escape(prof)}</div>{summary_html}</div>"
                                "</details>"
                            )
                        html += '</div>'
                    html += '</div>'
                    return html
                
                # Replace the update_buttons function - only show detected entities
                all_courses_state.change(update_buttons_with_handlers, [all_courses_state, all_professors_state, entity_summaries_state], entity_buttons_html)
                all_professors_state.change(update_buttons_with_handlers, [all_courses_state, all_professors_state, entity_summaries_state], entity_buttons_html)
                entity_summaries_state.change(update_buttons_with_handlers, [all_courses_state, all_professors_state, entity_summaries_state], entity_buttons_html)
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch(show_api=False)  # Hide API documentation
