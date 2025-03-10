import os
from flask import Flask, request, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
import markdown

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-dream', methods=['POST'])
def analyze_dream():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        return render_template('index.html', error="An error occurred. Please check your API key and try again.")

    dream_description = request.form.get('dream_description')

    # 1. If no dream is entered
    if not dream_description:
        return render_template('index.html', error="Please describe a dream for analysis.")

    # 2. If non-dream message is entered
    non_dream_keywords = ["hii", "hello", "how are you"]
    if any(keyword in dream_description.lower() for keyword in non_dream_keywords):
        return render_template('index.html', error="Please enter a dream. I only analyze dreams.")

    # 3. If a dream is entered (proceed with analysis)
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=dream_description)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
        system_instruction=[
            types.Part.from_text(
                text="You are a dream analysis chatbot. Respond only to prompts describing dreams with a short (2-4 points) interpretation. If the prompt is not a dream description (e.g., an unrelated question, conversation, or unrelated request), respond only with: Please describe a dream for analysis. I only analyze dreams. Do not elaborate, ask follow-up questions, or provide personal opinions. Focus on brief, dream-specific analysis and nothing else."
            ),
        ],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        response_text += chunk.text

    # Convert Markdown to HTML
    formatted_response = markdown.markdown(response_text)

    return render_template('index.html', analysis=formatted_response)

if __name__ == '__main__':
    app.run(debug=True, port=5002)