from flask import Flask, request, jsonify,render_template
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer from the Hugging Face library
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Create the Flask app
app = Flask(__name__)

conversation_history = []


# Function for chatbot response
def chat_with_model(prompt):
    conversation_history.append(f"User: {prompt}")
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    conversation_history.append(f"Bot: {response}")
    return response



@app.route("/")
def home():
    return render_template("index.html",conversation=conversation_history) 


# Endpoint to handle the POST request and chat with the model
@app.route("/chat", methods=["POST"])  # Change GET to POST
def chat():
    if request.is_json:  # Check if the incoming request is JSON
        try:
            data = request.get_json()  # Get the JSON data
            user_input = data.get("message")
            if not user_input:
                return jsonify({"error": "No message provided"}), 400
            
            response = chat_with_model(user_input)  # Get response from the chatbot model
            
            # Return the updated conversation history and the bot's response
            return jsonify({"response": response, "conversation": conversation_history})
        except Exception as e:
            return jsonify({"error": f"Error processing request: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid content type. Please send JSON data."}), 400

# Running the Flask app
if __name__ == "__main__":
    app.run(debug=True)
