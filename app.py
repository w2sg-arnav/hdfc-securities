from flask import Flask, request, jsonify

# chatbot = RAGCHATBOT()

# def process_messaget():

#     return final_message, accuracy

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json()
    user_message = data['message']
    # model_message, accuracy = chatbot(user_message)

    response = {
        # 'message': model_message,
        # 'accuracy': accuracy
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
