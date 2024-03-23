from django.shortcuts import render, HttpResponse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from data_scrap.views import fetchRealTimeData
from rest_framework import viewsets
from rest_framework.response import Response
from .models import RealTimePrediction
from .serializers import RealTimePredictionSerializer
from skimage.color import rgb2gray
from skimage.segmentation import chan_vese
from django.core.files.base import ContentFile
import io
import cv2
from PIL import Image
import random
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

loaded_model = tf.keras.models.load_model('model.h5')

def image_crop(image, filename):
    crop_coords = (200, 500, 1060, 1280)
    cropped_image = image.crop(crop_coords)
    return np.array(cropped_image)

def apply_DoG(image):
    image_uint8 = (image * 255).astype(np.uint8)
    blurred1 = cv2.GaussianBlur(image_uint8, (9, 9), 1.0)
    blurred2 = cv2.GaussianBlur(image_uint8, (3, 3), 0.5)
    dog_image = blurred1 - blurred2
    return dog_image.astype(np.float32) / 255

def image_segmentation_level_sets(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = chan_vese(img_gray, mu=0.25, lambda1=1, lambda2=1, tol=1e-3)
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = [255, 255, 255]
    return segmented_image

def load_and_preprocess_single_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_array = np.array(image)
        image_pil = Image.fromarray(image_array)

        cropped_image_pil = image_crop(image_pil, image_path)
        cropped_image_array = np.array(cropped_image_pil)

        segmented_image = image_segmentation_level_sets(cropped_image_array)

        image = tf.convert_to_tensor(segmented_image)

        image = tf.image.resize(image, (256, 256))
        image = image / 255.0
        return image
    except Exception as err:
        print(err)
        return None

def getOtherMetrics(wind_speed):
    if wind_speed <= 65:
        return 1000, 1, 0
    elif wind_speed <= 90:
        return 987, 4, 1
    elif wind_speed <= 102:
        return 970, 5, 2
    elif wind_speed <= 115:
        return 960, 5.5, 3
    elif wind_speed <= 140:
        return 948, 6, 4
    else:
        return 921, 7, 5   

def predictRealTime():
    global loaded_model
    #fetchRealTimeData()

    new_image_path = 'static/real-time-scrap/processing.png'
    original_image_path = 'static/real-time-scrap/processing.png'
    ir_image_path = 'static/real-time-scrap/ir.png'
    original_image = cv2.imread(original_image_path)
    original_image = original_image[500:1280, 200:1060]
    #original_image = original_image.crop((480, 0, 980, 500))
    ir_image = Image.open(ir_image_path)
    ir_image = ir_image.crop((480, 0, 980, 500))

    print("Original image shape:", original_image.shape)
    print("IR image shape:", ir_image.size)

    processed_image = load_and_preprocess_single_image(new_image_path)

    if processed_image is not None:
        processed_image = np.expand_dims(processed_image, axis=0)
        prediction = loaded_model.predict(processed_image)
        print(f'Predicted WMO_WIND: {prediction[0][0]}')

        original_buffer = io.BytesIO()
        original_pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        original_pil_image.save(original_buffer, format='PNG')
        original_buffer.seek(0)

        original_img = ContentFile(original_buffer.read(), name='original.png')

        ir_image_rgba = ir_image.convert("RGBA")
        ir_buffer = io.BytesIO()
        ir_image_rgba.save(ir_buffer, format='PNG')
        ir_buffer.seek(0)

        processed_img = ContentFile(ir_buffer.read(), name='processed.png')

        pressure, t_number, category = getOtherMetrics(float(prediction[0][0]))

        cyclone_intensity = RealTimePrediction.objects.create(
            wind=float(prediction[0][0]), 
            pressure=pressure+random.randint(0,5),
            t_number=t_number,
            category=category,
            original_img=original_img, 
            processed_img=processed_img,
        )
        cyclone_intensity.save()
        
        # Send real-time update to WebSocket clients
        # channel_layer = get_channel_layer()
        # async_to_sync(channel_layer.group_send)(
        #     "predictions", {"type": "send_prediction_update", "data": RealTimePredictionSerializer(cyclone_intensity).data}
        # )
    else:
        print("Image loading or preprocessing failed.")
    return HttpResponse("Done")

class RealTimePredictionViewSet(viewsets.ModelViewSet):
    serializer_class = RealTimePredictionSerializer
    queryset = RealTimePrediction.objects.all().order_by('id')

    def list(self, request):
        #predictRealTime()
        return Response(RealTimePredictionSerializer(RealTimePrediction.objects.order_by('-timestamp').first()).data)