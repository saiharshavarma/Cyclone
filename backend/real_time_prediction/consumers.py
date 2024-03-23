# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class PredictionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("predictions", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("predictions", self.channel_name)

    async def send_prediction_update(self, event):
        prediction_data = event["data"]
        await self.send(text_data=json.dumps(prediction_data))
