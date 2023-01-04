import contextlib
import json
import logging.config
import os
from datetime import datetime
import random

import requests
import time

from cltl.backend.api.backend import Backend
from cltl.backend.api.camera import CameraResolution, Camera
from cltl.backend.api.microphone import Microphone
from cltl.backend.api.storage import AudioStorage, ImageStorage
from cltl.backend.api.text_to_speech import TextToSpeech
from cltl.backend.impl.cached_storage import CachedAudioStorage, CachedImageStorage
from cltl.backend.impl.image_camera import ImageCamera
from cltl.backend.impl.sync_microphone import SynchronizedMicrophone
from cltl.backend.impl.sync_tts import SynchronizedTextToSpeech, TextOutputTTS
from cltl.backend.server import BackendServer
from cltl.backend.source.client_source import ClientAudioSource, ClientImageSource
from cltl.backend.source.console_source import ConsoleOutput
from cltl.backend.spi.audio import AudioSource
from cltl.backend.spi.image import ImageSource
from cltl.backend.spi.text import TextOutput
from cltl.chatui.api import Chats
from cltl.chatui.memory import MemoryChats
from cltl.combot.event.bdi import IntentionEvent
from cltl.combot.infra.config.k8config import K8LocalConfigurationContainer
from cltl.combot.infra.di_container import singleton
from cltl.combot.infra.event import Event
from cltl.combot.infra.event.memory import SynchronousEventBusContainer
from cltl.combot.infra.resource.threaded import ThreadedResourceContainer
from cltl.eliza.api import Eliza
from cltl.eliza.eliza import ElizaImpl
from cltl.gestures.gestures import GestureType
from cltl.vad.webrtc_vad import WebRtcVAD
from cltl_service.asr.service import AsrService
from cltl_service.backend.backend import BackendService
from cltl_service.backend.storage import StorageService
from cltl_service.chatui.service import ChatUiService
from cltl_service.eliza.service import ElizaService
from cltl_service.intentions.init import InitService
from cltl_service.vad.service import VadService
from flask import Flask
from hi.persbot.api import Persbot
from hi.persbot.persbot import PersbotImpl
from hi_service.persbot.service import PersbotService
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from cltl.emissordata.api import EmissorDataStorage
from cltl.emissordata.file_storage import EmissorDataFileStorage
from cltl_service.bdi.service import BDIService
from cltl_service.context.service import ContextService
from cltl_service.emissordata.client import EmissorDataClient
from cltl_service.emissordata.service import EmissorDataService
from cltl_service.keyword.service import KeywordService

from emissor.representation.util import serializer as emissor_serializer

logging.config.fileConfig('config/logging.config', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class InfraContainer(SynchronousEventBusContainer, K8LocalConfigurationContainer, ThreadedResourceContainer):
    def start(self):
        pass

    def stop(self):
        pass


class RemoteTextOutput(TextOutput):
    def __init__(self, remote_url: str):
        self._remote_url = remote_url

    def consume(self, text: str, language=None):
        tts_headers = {'Content-type': 'text/plain'}

        # animation = gestures.BOW
        animation = f"{random.choice(list(GestureType))}"
        print("THIS IS WHAT YOU SHOULD VERBALIZE FOR US:", text, animation)

        response = f"\\^startTag({animation}){text}^stopTag({animation})"  #### cannot pass in strings with quotes!!

        requests.post(f"{self._remote_url}/text", data=response, headers=tts_headers)


class BackendContainer(InfraContainer):
    @property
    @singleton
    def audio_storage(self) -> AudioStorage:
        return CachedAudioStorage.from_config(self.config_manager)

    @property
    @singleton
    def image_storage(self) -> ImageStorage:
        return CachedImageStorage.from_config(self.config_manager)

    @property
    @singleton
    def audio_source(self) -> AudioSource:
        return ClientAudioSource.from_config(self.config_manager)

    @property
    @singleton
    def image_source(self) -> ImageSource:
        return ClientImageSource.from_config(self.config_manager)

    @property
    @singleton
    def text_output(self) -> TextOutput:
        config = self.config_manager.get_config("cltl.backend.text_output")
        remote_url = config.get("remote_url")
        if remote_url:
            return RemoteTextOutput(remote_url)
        else:
            return ConsoleOutput()

    @property
    @singleton
    def microphone(self) -> Microphone:
        return SynchronizedMicrophone(self.audio_source, self.resource_manager)

    @property
    @singleton
    def camera(self) -> Camera:
        config = self.config_manager.get_config("cltl.backend.image")

        return ImageCamera(self.image_source, config.get_float("rate"))

    @property
    @singleton
    def tts(self) -> TextToSpeech:
        return SynchronizedTextToSpeech(TextOutputTTS(self.text_output), self.resource_manager)

    @property
    @singleton
    def backend(self) -> Backend:
        return Backend(self.microphone, self.camera, self.tts)

    @property
    @singleton
    def backend_service(self) -> BackendService:
        return BackendService.from_config(self.backend, self.audio_storage, self.image_storage,
                                          self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def storage_service(self) -> StorageService:
        return StorageService(self.audio_storage, self.image_storage)

    @property
    @singleton
    def server(self) -> Flask:
        if not self.config_manager.get_config('cltl.backend').get_boolean("run_server"):
            # Return a placeholder
            return ""

        audio_config = self.config_manager.get_config('cltl.audio')
        video_config = self.config_manager.get_config('cltl.video')

        return BackendServer(audio_config.get_int('sampling_rate'), audio_config.get_int('channels'),
                             audio_config.get_int('frame_size'),
                             video_config.get_enum('resolution', CameraResolution),
                             video_config.get_int('camera_index'))

    def start(self):
        logger.info("Start Backend")
        super().start()
        if self.server:
            self.server.start()
        self.storage_service.start()
        self.backend_service.start()

    def stop(self):
        logger.info("Stop Backend")
        self.storage_service.stop()
        self.backend_service.stop()
        if self.server:
            self.server.stop()
        super().stop()


class VADContainer(InfraContainer):
    @property
    @singleton
    def vad_service(self) -> VadService:
        config = self.config_manager.get_config("cltl.vad.webrtc")
        activity_window = config.get_int("activity_window")
        activity_threshold = config.get_float("activity_threshold")
        allow_gap = config.get_int("allow_gap")
        padding = config.get_int("padding")
        storage = None
        # DEBUG
        # storage = "/Users/tkb/automatic/workspaces/robo/eliza-parent/cltl-eliza-app/py-app/storage/audio/debug/vad"

        vad = WebRtcVAD(activity_window, activity_threshold, allow_gap, padding, storage=storage)

        return VadService.from_config(vad, self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start VAD")
        super().start()
        self.vad_service.start()

    def stop(self):
        logger.info("Stop VAD")
        self.vad_service.stop()
        super().stop()


class EmissorStorageContainer(InfraContainer):
    @property
    @singleton
    def emissor_storage(self) -> EmissorDataStorage:
        return EmissorDataFileStorage.from_config(self.config_manager)

    @property
    @singleton
    def emissor_data_service(self) -> EmissorDataService:
        return EmissorDataService.from_config(self.emissor_storage,
                                              self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def emissor_data_client(self) -> EmissorDataClient:
        return EmissorDataClient("http://0.0.0.0:8000/emissor")

    def start(self):
        logger.info("Start Emissor Data Storage")
        super().start()
        self.emissor_data_service.start()

    def stop(self):
        logger.info("Stop Emissor Data Storage")
        self.emissor_data_service.stop()
        super().stop()


class ASRContainer(EmissorStorageContainer, InfraContainer):
    @property
    @singleton
    def asr_service(self) -> AsrService:
        config = self.config_manager.get_config("cltl.asr")
        sampling_rate = config.get_int("sampling_rate")
        implementation = config.get("implementation")

        storage = None
        # DEBUG
        # storage = "/Users/tkb/automatic/workspaces/robo/eliza-parent/cltl-eliza-app/py-app/storage/audio/debug/asr"

        if implementation == "google":
            from cltl.asr.google_asr import GoogleASR
            impl_config = self.config_manager.get_config("cltl.asr.google")
            asr = GoogleASR(impl_config.get("language"), impl_config.get_int("sampling_rate"),
                            hints=impl_config.get("hints", multi=True))
        elif implementation == "speechbrain":
            from cltl.asr.speechbrain_asr import SpeechbrainASR
            impl_config = self.config_manager.get_config("cltl.asr.speechbrain")
            model = impl_config.get("model")
            asr = SpeechbrainASR(model, storage=storage)
        elif implementation == "wav2vec":
            from cltl.asr.wav2vec_asr import Wav2Vec2ASR
            impl_config = self.config_manager.get_config("cltl.asr.wav2vec")
            model = impl_config.get("model")
            asr = Wav2Vec2ASR(model, sampling_rate=sampling_rate, storage=storage)
        elif not implementation:
            asr = False
        else:
            raise ValueError("Unsupported implementation " + implementation)

        if asr:
            return AsrService.from_config(asr, self.emissor_data_client,
                                          self.event_bus, self.resource_manager, self.config_manager)
        else:
            logger.warning("No ASR implementation configured")
            return False

    def start(self):
        super().start()
        if self.asr_service:
            logger.info("Start ASR")
            self.asr_service.start()

    def stop(self):
        if self.asr_service:
            logger.info("Stop ASR")
            self.asr_service.stop()
        super().stop()


class ElizaComponentsContainer(EmissorStorageContainer, InfraContainer):
    @property
    @singleton
    def keyword_service(self) -> KeywordService:
        return KeywordService.from_config(self.emissor_data_client,
                                          self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def context_service(self) -> ContextService:
        return ContextService.from_config(self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def keyword_service(self) -> KeywordService:
        return KeywordService.from_config(self.emissor_data_client,
                                          self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def bdi_service(self) -> BDIService:
        # TODO make configurable
        bdi_model = {"init":
                         {"initialized": ["persbot"]},
                     "persbot":
                         {"quit": ["init"]}
                     }

        return BDIService.from_config(bdi_model, self.event_bus, self.resource_manager, self.config_manager)

    @property
    @singleton
    def init_intention(self) -> InitService:
        return InitService.from_config(self.emissor_data_client,
                                       self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Eliza services")
        super().start()
        self.bdi_service.start()
        self.keyword_service.start()
        self.context_service.start()
        self.init_intention.start()

    def stop(self):
        logger.info("Stop Eliza services")
        self.init_intention.stop()
        self.bdi_service.stop()
        self.keyword_service.stop()
        self.context_service.stop()
        super().stop()


class ChatUIContainer(InfraContainer):
    @property
    @singleton
    def chats(self) -> Chats:
        return MemoryChats()

    @property
    @singleton
    def chatui_service(self) -> ChatUiService:
        return ChatUiService.from_config(MemoryChats(), self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Chat UI")
        super().start()
        self.chatui_service.start()

    def stop(self):
        logger.info("Stop Chat UI")
        self.chatui_service.stop()
        super().stop()


class ElizaContainer(EmissorStorageContainer, InfraContainer):
    @property
    @singleton
    def eliza(self) -> Eliza:
        return ElizaImpl()

    @property
    @singleton
    def eliza_service(self) -> ElizaService:
        return ElizaService.from_config(self.eliza, self.emissor_data_client,
                                        self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Eliza")
        super().start()
        self.eliza_service.start()

    def stop(self):
        logger.info("Stop Eliza")
        self.eliza_service.stop()
        super().stop()


class PersbotContainer(EmissorStorageContainer, InfraContainer):
    @property
    @singleton
    def persbot(self) -> Persbot:
        return PersbotImpl()

    @property
    @singleton
    def persbot_service(self) -> PersbotService:
        return PersbotService.from_config(self.persbot, self.emissor_data_client,
                                        self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Persbot")
        super().start()
        self.persbot_service.start()

    def stop(self):
        logger.info("Stop Eliza")
        self.persbot_service.stop()
        super().stop()


class ApplicationContainer(PersbotContainer, ElizaComponentsContainer,
                           ChatUIContainer,
                           ASRContainer, VADContainer,
                           EmissorStorageContainer, BackendContainer):
    pass


def get_event_log_path(config):
    log_dir = config.get('event_log')
    date_now = datetime.now()

    os.makedirs(log_dir, exist_ok=True)

    return f"{log_dir}/{date_now :%y_%m_%d-%H_%M_%S}.json"


@contextlib.contextmanager
def event_log(event_bus, config):
    def log_event(event):
        try:
            event_log.write(json.dumps(event, default=serializer, indent=2) + ',\n')
        except:
            logger.exception("Failed to write event: %s", event)

    with open(get_event_log_path(config), "w") as event_log:
        event_log.writelines(['['])

        topics = event_bus.topics
        for topic in topics:
            event_bus.subscribe(topic, log_event)
        logger.info("Subscribed %s to %s", event_log.name, topics)

        yield None

        event_log.writelines([']'])


def serializer(obj):
    try:
        return emissor_serializer(obj)
    except Exception:
        try:
            return vars(obj)
        except Exception:
            return str(obj)


def main():
    ApplicationContainer.load_configuration()

    logger.info("Initialized Application")

    application = ApplicationContainer()
    application.start()

    intention_topic = application.config_manager.get_config("cltl.bdi").get("topic_intention")
    application.event_bus.publish(intention_topic, Event.for_payload(IntentionEvent(["init"])))

    config = application.config_manager.get_config("cltl.leolani")
    with event_log(application.event_bus, config):
        routes = {
            '/storage': application.storage_service.app,
            '/emissor': application.emissor_data_service.app,
            '/chatui': application.chatui_service.app
        }

        if application.server:
            routes['/host'] = application.server.app

        web_app = DispatcherMiddleware(Flask("Eliza app"), routes)

        run_simple('0.0.0.0', 8000, web_app, threaded=True, use_reloader=False, use_debugger=False, use_evalex=True)

        intention_topic = application.config_manager.get_config("cltl.bdi").get("topic_intention")
        application.event_bus.publish(intention_topic, Event.for_payload(IntentionEvent(["terminate"])))
        time.sleep(1)

        application.stop()


if __name__ == '__main__':
    main()