import os
from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
from google import genai
from google.genai import types
import markdown

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-dream', methods=['POST'])
def analyze_dream():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        return render_template('index.html', error="An error occurred. Please check your API key and try again.")

    dream_description = request.form.get('dream_description')
    if not dream_description:
        return render_template('index.html', error="Please describe a dream for analysis.")

    # Check for non-dream input
    non_dream_keywords = ["hii", "hello", "how are you"]
    if any(keyword in dream_description.lower() for keyword in non_dream_keywords):
      return render_template('index.html', error="I only analyze dreams. Please describe a dream for analysis.")

   
    
    
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
                threshold="BLOCK_NONE",
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
                text="You are a dream analysis chatbot. Please analyze any user-described dream, even if it contains violence or conflict. Provide a brief (2-4 point) interpretation focused only on the dream’s meaning."
            ),
        ],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        response_text += chunk.text
    # 🔍 Debug Logging
    print("Dream Description:", dream_description)
    print("Generated Response:", response_text)
    # 🛑 Fallback if the response is not valid
    if not response_text or "Please describe a dream for analysis" in response_text:
      return render_template('index.html', error="Sorry, we couldn't analyze the dream. Please try again with a different description.")
    # Convert Markdown to HTML
    formatted_response = markdown.markdown(response_text)

    return redirect(url_for('result', analysis=formatted_response))

@app.route('/result')
def result():
    analysis = request.args.get('analysis')
    return render_template('result.html', analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
