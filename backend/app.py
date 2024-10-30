import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from asl_recognition.model import recognize_asl
from db_manager import db_manager

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.errorhandler(Exception) # type: ignore
def handle_exception(e):
    if isinstance(e, HTTPException):
        response = e.get_response()
        response_data = {
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }
        return jsonify(response_data), e.code

    logger.exception("An unexpected error occurred")
    return jsonify({
        "code": 500,
        "name": "Internal Server Error",
        "description": "An unexpected error occurred",
    }), 500

@app.route('/translate', methods=['POST'])
def translate_asl():
    try:
        if 'video' not in request.files:
            raise ValueError("No video provided")

        video = request.files['video']
        user_id = request.form.get('user_id')

        asl_text = recognize_asl(video)
        translation_id = db_manager.store_translation(user_id, asl_text)

        return jsonify({
            "translation_id": translation_id,
            "asl_text": asl_text
        })

    except ValueError as ve:
        logger.warning(f"Invalid input: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("An error occurred during translation")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)