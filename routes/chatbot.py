from flask import Flask, request, jsonify, make_response, Blueprint
import redis
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chatbot_bp = Blueprint('chatbot', __name__)


model = ChatOpenAI(model="gpt-4o-mini")

redis_client = redis.Redis(
    host=os.getenv("REDIS_URL"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USER_NAME"),
    password=os.getenv("REDIS_API_PASSWORD")
)

def get_memory(session_id):
    """Retrieve the summarized conversation from Redis as a list."""
    summary = redis_client.get(f"summary:{session_id}")
    if summary:
        try:
            return json.loads(summary)
        except Exception as e:
            print("Error loading memory:", e)
    return []


def save_memory(session_id, conversation_list):
    """Store the updated conversation list in Redis."""
    redis_client.set(f"summary:{session_id}", json.dumps(conversation_list))


@chatbot_bp.route("/v1/check-instruction", methods=["POST"])
def is_instruction_request():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")

    # conversation_history = get_memory(session_id)

    messages = [
        {"role": "system", "content": f""" You are an AI assistant analyzing a conversation where a user is designing or modifying an AI assistant.

                                        Determine if the latest message shows that the user is shaping, refining, or modifying the assistant's behavior, purpose, or instructions.

                                        Latest User Message:
                                        "{user_input}"

                                        Respond with:
                                        - yes — if the user is creating, modifying, asking about, or clarifying how the assistant should behave or what it should do/avoid.
                                        - no — if the message is unrelated to defining the assistant's function (e.g., general questions, small talk, unrelated tasks).

                                        Only reply with "yes" or "no". Do not include quotes or additional text."""
         }
    ]

    response = model.invoke(messages, temperature=0)
    classification = response.content.strip().lower()
    return jsonify({
        "response": classification == "yes"
    })  # classification = response.content.strip().lower()
    #
    # return classification == "yes"


@chatbot_bp.route("/v1/update-instruction", methods=["POST"])
def update_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    existing_instruction = data.get("existing_instruction")
    new_info = data.get("new_instruction")

    # conversation_history = get_memory(session_id)

    # print(new_info)

    prompt = f"""
    Update the following assistant description to naturally include the instruction: "{user_input}"

    Keep the original structure, tone, and formatting. Only modify where necessary to include the new instruction smoothly.

    Return only the updated description, with no extra explanation or formatting.

    Assistant description:

    \"\"\"{existing_instruction}\"\"\"
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    update_response = model.invoke(messages, temperature=0)
    print(update_response)

    # update_response = model.invoke(messages_update, temperature=0)
    updated_instruction = update_response.content.strip('""""')
    return updated_instruction


@chatbot_bp.route("/v1/get-instruction", methods=["POST"])
def get_ai_generated_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    # 🔹 Retrieve conversation history from Redis
    # conversation_history = get_memory(session_id)

    new_instruction = generate_ai_system_prompt(user_input)

    ai_suggested_name = suggest_profile_name(new_instruction)

    return jsonify({
        "new_instruction": new_instruction,
        "ai_suggested_description": ai_suggested_name
    })


def suggest_profile_name(instruction):
    """Suggests a meaningful assistant name based on conversation context."""
    messages = [
        {"role": "system", "content": """
            Generate a **concise, relevant, and descriptive** phrase summarizing the AI assistant's purpose. 

            **Guidelines:**
            - Extract the **core function** from the given instruction.
            - Keep the description **short (less than 15 words)**.
            - Ensure it sounds **natural and professional**.
            - Avoid generic terms like "AI Assistant" and instead describe **what the assistant does**.

            **Example Outputs:**
            - Input: "Make a creative writing assistant"
              Output: "A creative writing assistant for brainstorming and refining stories."
            - Input: "Make a software engineer assistant"
              Output: "A software engineer assistant for formatting and refining code."

            Now, generate a short description for the following instruction.
        """},
        {"role": "user", "content": str(instruction)}
    ]

    response = model.invoke(messages, temperature=0.5)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data
    except Exception as e:
        print(f"Error suggesting assistant name: {e}")
        return "AI Assistant"  # Fallback name


def generate_ai_system_prompt(user_prompt):
    messages = [
        {"role": "system",
         "content": "You are an AI that generates concise, structured, and natural-sounding instructions for a Custom GPT."},
        {"role": "system", "content": f"""
            Generate a **short, structured, and engaging** instruction for a GPT assistant in a natural tone.

            **Format the response as follows:**
            - Begin with: **"This OOm Writing Assistant is a..."** followed by a **clear, direct description** of what it does.
            - Follow with **a sentence or two** explaining how it interacts with users in a supportive and engaging way.
            - End with a **short, smooth statement** about what it avoids, ensuring flexibility while setting constraints.

            The response should be **brief, expressive, and fluid**—avoiding section titles, lists, or rigid formatting.

            Now, generate an instruction for a GPT that serves as a {user_prompt}.
        """},
    ]

    try:
        response = model.invoke(messages, temperature=0.7)
        refined_instruction = response.content.strip()
        return refined_instruction

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error generating refined instruction."


def update_title(updated_instruction):
    title_messages = [
        {"role": "system", "content": """
                                            Analyze the following AI assistant description and extract a **concise assistant profile name**.

                                            **Requirements:**
                                            - The name should be **clear, relevant, and meaningful**.
                                            - **Do NOT exceed 50 characters**.
                                            - Keep it professional and structured.
                                            - Do NOT include unnecessary words like "This GPT is...".
                                            - Return ONLY a valid JSON response without any extra text or formatting.

                                            Output format:
                                            {
                                                "assistantProfile": ""
                                            }
                                        """
         },
        {"role": "user", "content": updated_instruction}
    ]
    title_response = model.invoke(title_messages, temperature=0)
    extracted_title = title_response.content.strip()

    return extracted_title


def safe_openai_call(messages, temperature=0.7):
    """Safely invoke OpenAI API with error handling."""
    try:
        response = model.invoke(messages, temperature=temperature)
        return response.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I'm experiencing technical issues. Please try again later."


def extract_assistant_name(conversation_history):
    """Extract a meaningful assistant name based on previous interactions."""
    messages = [
        {"role": "system", "content": """
            Analyze the following conversation and determine the **best possible name** for the AI assistant.

            **Guidelines:**
            - The name should be **clear, relevant, and professional**.
            - **Do NOT exceed 50 characters**.
            - Avoid generic names like "AI" or "Assistant" alone.
            - Keep it simple yet descriptive (e.g., "AI Writing Assistant").
            - Return only a **valid JSON object**.

            **Output format:**
            { "assistantProfile": "Assistant Name Here" }
        """},
        {"role": "user", "content": str(conversation_history)}
    ]
    response = model.invoke(messages, temperature=0)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data.get("assistantProfile", "AI Assistant")  # Default fallback
    except Exception as e:
        print(f"Error extracting assistant name: {e}")
        return "AI Assistant"  # Fallback name


def update_conversation_history(session_id, user_input, ai_response):
    """Update conversation history and prevent duplicates."""
    conversation_history = get_memory(session_id)

    if conversation_history:
        last_entry = conversation_history[-1]
        if last_entry.get("role") == "assistant" and last_entry.get("content") == ai_response:
            return  # Avoid duplicate AI responses

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": ai_response})

    save_memory(session_id, conversation_history)


# Using
def extract_custom_name(user_input):
    messages = [
        {"role": "system", "content": """ 
            Extract the assistant name from the user's message.

            **Instructions:**
            - The user might provide a name directly, such as:
              - "I prefer BrainyBot."
              - "Let’s call it AI Master."
              - "How about Pro Max Specialist?"
            - Only return the **extracted name** without any extra text.

            **Example Outputs:**
            - **User:** "I prefer BrainyBot." → **Response:** `"BrainyBot"`
            - **User:** "Let’s name it Pro Max Specialist." → **Response:** `"Pro Max Specialist"`
            - **User:** "I like AI Genius." → **Response:** `"AI Genius"`

            Do NOT include extra characters, explanations, or formatting or "" or single quote
        """},
        {"role": "user", "content": user_input}
    ]

    response = model.invoke(messages, temperature=0)
    return response.content.strip()


# Using
def suggest_assistant_name(conversation_history):
    """Suggests a meaningful assistant name based on conversation context."""
    messages = [
        {"role": "system", "content": """
            Analyze the following conversation and suggest the **best possible name** for the AI assistant.

            **Guidelines:**
            - The name should be **clear, relevant, and professional**.
            - **Do NOT exceed 50 characters**.
            - Avoid generic names like "AI" or "Assistant" alone.
            - Suggest something meaningful based on the context.
            - Keep it simple yet descriptive (e.g., "AI Writing Assistant").
            - Return only a **valid JSON object**.

            **Output format:**
            { "suggestedName": "Assistant Name Here" }
        """},
        {"role": "user", "content": str(conversation_history)}
    ]

    response = model.invoke(messages, temperature=0)

    try:
        extracted_data = json.loads(response.content.strip())  # Ensure valid JSON
        return extracted_data.get("suggestedName", "AI Assistant")  # Default fallback
    except Exception as e:
        print(f"Error suggesting assistant name: {e}")
        return "AI Assistant"  # Fallback name


# Using
def detect_name_confirmation(conversation_history, user_input):
    print(conversation_history)
    messages = [
        {"role": "system", "content": """ 
                You are analyzing user responses to determine if they are confirming, rejecting, or changing the topic.

                **Instructions:**
                - If the user confirms (e.g., "That works!", "Sounds great", "I like it"), respond with `"confirmed"`.
                - If the user rejects and wants a new name (e.g., "No", "Nah, another one", "I don’t like that"), respond with `"rejected"`.
                - If the user suggests their own name (e.g., "I prefer BrainyBot", "Let’s call it AI Master"), respond with `"custom_name"`.
                - If the user has changed the topic (e.g., "Tell me about SEO", "How do I use AI for writing?"), respond with `"topic_change"`.

                **Response format:**
                Return only one of the following exact words:
                - `"confirmed"`
                - `"rejected"`
                - `"custom_name"`
                - `"topic_change"`

                Do NOT include extra characters, explanations, or formatting.
            """}
    ]

    # Include old conversation history to provide context
    if conversation_history:
        messages.extend(conversation_history)

    # Add the latest user message
    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0.1)

    return response.content.strip()


@chatbot_bp.route("/v1/test-instruction", methods=["POST"])
def test_instruction():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    name = data.get("name")
    tone_of_voice_and_vocab = data.get("tone_of_voice_and_vocab")
    target_audience = data.get("target_audience")
    important_notes = data.get("important_notes")
    conversation_history = get_memory(session_id)

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    system_instruction = (
        "You are a blog-writing AI assistant. The user will provide information step by step "
        "(e.g., topic, tone, target audience, word count). "
        "Do not generate the blog until the user clearly asks you to. "
        "Instead, ask clarifying questions or confirm details if needed."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        SystemMessage(content=f"Current Assistant Name: {name}"),
        SystemMessage(content= f"Tone of Voice and Vocabulary: {tone_of_voice_and_vocab}\n"
                                                                f"Target Audience: {target_audience}\n"
                                                                f"Important Notes: {important_notes}"),
        MessagesPlaceholder(variable_name="conversation_history"),
        HumanMessage(content=user_input)
    ])

    response = prompt_template | model
    ai_response = response.invoke(
        {"conversation_history": conversation_history})

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": ai_response.content})
    save_memory(session_id, conversation_history)

    return jsonify({"response": ai_response.content.strip()})


@chatbot_bp.route("/v1/input-classification", methods=["POST"])
def input_classification():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    conversation_history = get_memory(session_id)

    messages = [
        {
            "role": "system",
            "content": """
                            You are an AI assistant analyzing a conversation to determine the user's intent.

                            **Your task:** Based on the conversation so far and the user's latest message, classify what the user is trying to do.

                            **Possible classifications:**
                            - name_update: The user wants to rename the assistant or give it a new name.
                            - instruction_update: The user is trying to change, update, or define how the assistant behaves, responds, or what capabilities it has.
                            - chit_chat: The user is engaging in casual talk, greetings, or unrelated small talk.
                            - unclear: The message is vague, incomplete, or doesn't clearly match any of the above categories.

                            **Guidelines:**
                            - Use the latest message to infer intent.
                            - Use prior conversation only if the latest message is ambiguous.
                            - If the user is giving a task, prompt, or desired behavior (e.g., "I want to create...", "Make it respond like...", "It should..."), classify as instruction_update.
                            - Do not interpret or explain. Just classify.

                            **Output format:**
                            Return only one of the classification strings: name_update, instruction_update, chit_chat, or unclear

                            Do NOT include quotes, punctuation, or extra text.
                        """
        }
    ]
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0)

    classification = response.content.strip().lower()

    conversation_history.append({"role": "user", "content": user_input})
    # conversation_history.append({"role": "assistant", "content": classification})
    save_memory(session_id, conversation_history)
    return jsonify({
        "response": classification
    })


@chatbot_bp.route("/v1/ai-extract-name", methods=["POST"])
def ai_extract_name():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    is_name_confirmation = data.get("is_name_confirmation")
    suggested_name = data.get("suggested_name")
    conversation_history = get_memory(session_id)

    custom_name = extract_custom_name(user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": custom_name})
    save_memory(session_id, conversation_history)

    return jsonify(
        {"response": "custom_name", "detect_name_confirmation": False, "suggested_name": custom_name}), 200


@chatbot_bp.route("/v1/ai-name-confirmation", methods=["POST"])
def get_ai_name_confirmation():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    is_name_confirmation = data.get("is_name_confirmation")
    suggested_name = data.get("suggested_name")
    conversation_history = get_memory(session_id)

    if isinstance(is_name_confirmation, str):
        is_name_confirmation = is_name_confirmation.lower() == "true"

    if is_name_confirmation:
        detect_confirmation = detect_name_confirmation(conversation_history, user_input).strip().replace('"',
                                                                                                         '').lower()
        print(detect_confirmation)
        if detect_confirmation == "confirmed":
            return jsonify(
                {"response": "confirmed", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200

        elif detect_confirmation == "rejected":
            return jsonify(
                {"response": "rejected", "detect_name_confirmation": True, "suggested_name": suggested_name}), 200

        elif detect_confirmation == "topic_change":
            return jsonify(
                {"response": "change", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200
        elif detect_confirmation == "custom_name":
            custom_name = extract_custom_name(user_input)
            return jsonify(
                {"response": "custom_name", "detect_name_confirmation": False, "suggested_name": custom_name}), 200

    conversation_history.append({"role": "user", "content": user_input})
    save_memory(session_id, conversation_history)

    return jsonify(
        {"response": "no_confirmation", "detect_name_confirmation": False, "suggested_name": suggested_name}), 200


@chatbot_bp.route("/v1/ai-name-suggestion", methods=["POST"])
def get_ai_name_suggestion():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")

    conversation_history = get_memory(session_id)
    print(str(conversation_history))
    messages = [
        {"role": "system", "content": f"""
                   Analyze the following conversation and suggest the **best possible name** for the AI assistant.

                   **Guidelines:**
                   - Describe the assistant type in **3-5 words** (e.g., "AI Writing Assistant", "SEO Optimization Bot").
                   - Suggest a **creative, unique name** based on the assistant's purpose.
                   - **Avoid previously suggested names.
                   - Do NOT repeat past suggestions.
                   - Ensure the name is **concise and meaningful**.

                   **Return only the assistant name, no extra text.**
           """},
        {"role": "user", "content": str(conversation_history)}
    ]

    assistant_type = model.invoke(messages, temperature=0.9).content.strip()

    suggested_name = suggest_assistant_name(user_input)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": assistant_type})
    save_memory(session_id, conversation_history)
    # suggested_name = "Software Engineer"
    # redis_client.set(f"pending_name:{session_id}", suggested_name)

    return jsonify({
        "response": f"""Now, let's give this Assistant a name. How about **{suggested_name}**? Let me know if you have a different preference!""",
        "suggested_name": suggested_name
    })


@chatbot_bp.route("/v1/test-chat", methods=["POST"])
def test_chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    info = data.get("info")
    conversation_history = get_memory(session_id)

    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    messages = [
        # {"role": "system", "content": f"""{info}"""},
        {"role": "system", "content": f"""You are an **Intelligent Writing Assistant** that helps users define their **Writing Style**.

                    ## **Step 1: Ask Key Questions**
                    - Start with **three core questions**:
                      1. **What should the assistant do?**  
                         (Examples: Write concisely, provide detailed explanations, use storytelling, persuasive tone.)
                      2. **What should the assistant avoid?**  
                         (Examples: Jargon, passive voice, overly complex words, unnecessary details.)
                      3. **How should the assistant behave?**  
                         (Examples: Be proactive with suggestions, strictly follow rules, allow creativity and flexibility.)

                    - If the user is unsure, **offer predefined choices** to help them decide.

                    ## **Step 2: Summarize & Confirm**
                    - Summarize their preferences in a structured **Writing Style Guide**"""}
    ]

    response = model.invoke(messages, temperature=0.8)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response.content})
    save_memory(session_id, conversation_history)

    return jsonify({
        "response": response.content
    })


def refine_instruction(user_input):
    # Make a request to OpenAI for improving the instruction
    prompt = f"""
                   You are a system responsible for transforming user requests into well-structured AI assistant descriptions.

    The user has provided the following instruction for their AI assistant:
    "${user_input}"

    **Task:** Rewrite this instruction into a clear, well-structured AI system prompt while preserving its original intent and meaning.

    **Guidelines:**
    - **Ensure the description starts with "This Bot is..."**
    - Ensure the description is professional and coherent.
    - Expand relevant details if necessary to improve clarity.
    - Do **not** add unnecessary explanations or assumptions.
    - Make it structured but not overly robotic.

    **Output format:** Only return the structured instruction text, without additional formatting, examples, or explanations.
                """

    try:
        # Use the 'invoke' method from langchain's ChatOpenAI instead of Completion.create
        response = model.invoke([
            {"role": "system", "content": prompt}
        ], temperature=0.7)

        refined_instruction = response.content.strip()
        return refined_instruction

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error generating refined instruction."


@chatbot_bp.route("/v1/set-initial", methods=["POST"])
def set_initial_instruction():
    data = request.json
    instruction = data.get("instruction")
    name = data.get("name")
    description = data.get("description")
    session_id = data.get("session_id", "default_session")
    conversation_history = get_memory(session_id)
    print("Initial Session Id", session_id)

    if description:
        conversation_history.append({"role": "user", "content": f"name: {name}."})
    if name:
        conversation_history.append({"role": "user", "content": f"description: {description}"})
    if instruction:
        conversation_history.append({"role": "user", "content": instruction})

    save_memory(session_id, conversation_history)
    return jsonify({
        "response": "success"
    })


@chatbot_bp.route("/v1/ai-chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id", "default_session")
    instruction = data.get("instruction")
    conversation_history = get_memory(session_id)

    messages = [
        # {"role": "system", "content": f"""{instruction}"""},
        {"role": "system", "content": f"""
        You are a helpful and friendly assistant designed to guide users as they write or refine instructions for an AI assistant.

        Your role is to support the user step by step, keeping the conversation clear and productive.

        Guidelines:
        - If the user seems unsure or confused, suggest one or two specific next steps in a short, supportive response.
        - If the user's request is vague, ask one clear, focused follow-up question instead of guessing their intent.
        - Keep all responses short, conversational, and action-oriented to help the user move forward.
        - Do not explain concepts unless the user asks for clarification.
        - Always base your reply on the latest user message and the flow of the conversation.
        """}
    ]

    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    response = model.invoke(messages, temperature=0.3)

    # conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response.content})
    save_memory(session_id, conversation_history)

    print("AI Chat", conversation_history)
    return jsonify({
        "response": response.content
    })