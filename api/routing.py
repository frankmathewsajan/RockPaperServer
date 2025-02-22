from django.urls import re_path
from api.consumers import SignalingConsumer

websocket_urlpatterns = [
    re_path(r'ws/signaling/$', SignalingConsumer.as_asgi()),
]
