import pytest
from flask import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_translate_endpoint_no_image(client):
    response = client.post('/translate')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No image provided'

def test_translate_endpoint_with_image(client, mocker):
    mocker.patch('asl_recognition.model.recognize_asl', return_value='HELLO')
    mocker.patch('database.db_manager.DatabaseManager.store_translation', return_value=1)

    data = {
        'image': (io.BytesIO(b'test'), 'test.jpg'),
        'user_id': 'test_user'
    }
    response = client.post('/translate', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'translation_id' in data
    assert 'asl_text' in data
    assert data['asl_text'] == 'HELLO'