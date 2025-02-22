import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging

logger = logging.getLogger(__name__)


class SignalingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # In this simple example, all clients join the same room.
        self.room_group_name = 'webrtc_signaling'
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        logger.info("WebSocket signaling connection accepted. Channel: %s", self.channel_name)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        logger.info("WebSocket signaling connection closed. Channel: %s", self.channel_name)

    async def receive(self, text_data):
        logger.debug("Received signaling message: %s", text_data)
        # Broadcast the signaling message to all other participants in the room.
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'signaling_message',
                'message': text_data,
                'sender_channel': self.channel_name,
            }
        )

    async def signaling_message(self, event):
        message = event['message']
        sender_channel = event['sender_channel']
        # Avoid echoing the message back to the sender.
        if self.channel_name == sender_channel:
            return
        logger.debug("Sending signaling message to channel %s: %s", self.channel_name, message)
        await self.send(text_data=message)
