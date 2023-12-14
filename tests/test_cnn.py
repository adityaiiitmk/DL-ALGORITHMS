import pytest,cv2
import tensorflow as tf
from CNN.tumor_detection.cnn_tumor import make_prediction

@pytest.fixture
def sample_model():
    model_path = 'CNN/tumor_detection/results/model/cnn_tumor.h5'
    model = tf.keras.models.load_model(model_path)
    return model

@pytest.fixture
def sample_tumor_image():
    img_path = 'tests/samples/y7.jpg'
    img = cv2.imread(img_path)
    return img

@pytest.fixture
def sample_no_tumor_image():
    # Assuming you have a sample no tumor image for testing
    img_path = 'tests/samples/no11.jpg'
    img = cv2.imread(img_path)
    return img


def test_non_make_prediction (sample_model, sample_no_tumor_image):
    result = make_prediction(sample_no_tumor_image, sample_model)
    assert result == 0
    
def test_non_make_prediction (sample_model, sample_tumor_image):
    result = make_prediction(sample_tumor_image, sample_model)
    assert result == 1


