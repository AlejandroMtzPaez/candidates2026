import json
import queue
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from vision_interfaces.msg import FaceDetection
from vision_interfaces.srv import RagQuery

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model

SAMPLE_RATE = 16000
FRAME = 512                      # 32 ms, tamaño de ventana que espera Silero VAD
FRAME_MS = 1000 * FRAME / SAMPLE_RATE


def banco_mel(n_mels=40, n_fft=512, sr=SAMPLE_RATE, fmin=60, fmax=7600):
    mel = lambda f: 2595 * np.log10(1 + f / 700)          # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)          # noqa: E731
    puntos = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * puntos / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[i - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


MEL_FB = banco_mel()
VENTANA = np.hamming(400).astype(np.float32)


def huella_voz(audio):
    """Firma espectral ligera del hablante (log-mel promedio de las tramas con más energía).

    No es un x-vector, pero distingue razonablemente bien dos voces distintas
    (p. ej. tono grave vs. agudo) sin descargar modelos extra.
    """
    if len(audio) < 400 * 4:
        return None
    tramas = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160] * VENTANA
    espectro = np.abs(np.fft.rfft(tramas, 512)) ** 2
    logmel = np.log(espectro @ MEL_FB.T + 1e-8)
    energia = logmel.mean(axis=1)
    voz = logmel[energia >= np.percentile(energia, 50)]
    v = voz.mean(axis=0)
    v = v - v.mean()                       # quita la ganancia del micrófono
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_og')

        p = self.declare_parameter
        self.device_index = p('device_index', -1).value          # -1 = micrófono por defecto
        self.vad_on = p('vad_threshold', 0.5).value              # prob. para iniciar voz
        self.vad_off = p('vad_off_threshold', 0.35).value        # prob. para considerar silencio
        self.min_silence_ms = p('min_silence_ms', 600).value     # pausa que cierra un turno
        self.min_speech_ms = p('min_speech_ms', 300).value       # turnos más cortos se descartan
        self.max_turn_s = p('max_turn_s', 15.0).value
        self.speaker_threshold = p('speaker_similarity', 0.75).value
        self.max_speakers = p('max_speakers', 2).value
        self.echo_tail_s = p('echo_tail_s', 0.6).value
        self.face_timeout_s = p('face_timeout_s', 1.0).value

        # evita eco: voz.py publica /is_speaking mientras reproduce audio
        self.ignore_echo = False
        self.fin_eco = 0.0
        self.turno_ocupado = threading.Event()
        self.robot_hablando = False
        self.create_subscription(Bool, '/is_speaking', self.speaking_callback, 10)
        self.ultima_respuesta_robot = ''

        # face detection: rostros visibles por track_id
        self.lock = threading.Lock()
        self.rostros = {}               # track_id -> dict(name, area, t)
        self.speaker_a_track = {}       # 'S1' -> track_id
        self.create_subscription(FaceDetection, '/vision/detections', self.face_callback, 10)

        # diarización ligera: centroides de huella de voz por hablante
        self.centroides = []

        # rag y voz
        self.rag_client = self.create_client(RagQuery, 'consultar_rag')
        self.tts_pub = self.create_publisher(String, '/robot_response', 10)
        self.transcripcion_pub = self.create_publisher(String, '/transcription', 10)
        # avisa a face_detector cuándo se está procesando una respuesta, para que baje su ritmo
        self.ocupado_pub = self.create_publisher(Bool, '/robot_ocupado', 10)
        self.create_timer(0.2, lambda: self.ocupado_pub.publish(Bool(data=self.turno_ocupado.is_set())))

        # modelos
        self.get_logger().info('Cargando modelo Whisper...')
        self.model = WhisperModel('small', device='cpu', compute_type='int8')
        self.vad = get_vad_model()      # Silero VAD (ONNX) incluido en faster-whisper

        # Cola de turnos: un solo hilo transcribe, así los turnos nunca colisionan
        self.turnos = queue.Queue(maxsize=1)
        threading.Thread(target=self.listen_loop, daemon=True).start()
        threading.Thread(target=self.transcribe_loop, daemon=True).start()

    # ==========================================
    # CALLBACKS
    # ==========================================
    def face_callback(self, msg):
        area = max(0, msg.x2 - msg.x1) * max(0, msg.y2 - msg.y1)
        with self.lock:
            # si el track ya fue reconocido, un frame "Desconocido" no le borra el nombre
            nombre = msg.name
            anterior = self.rostros.get(msg.track_id, {}).get('name', 'Desconocido')
            if nombre == 'Desconocido' and anterior != 'Desconocido':
                nombre = anterior
            self.rostros[msg.track_id] = {'name': nombre, 'area': area, 't': time.time()}

    def speaking_callback(self, msg):
        if msg.data and not self.ignore_echo:
            self.get_logger().info('Robot hablando. Descarta el audio.')
        if msg.data:
            self.robot_hablando = True
        if not msg.data and self.ignore_echo:
            self.fin_eco = time.time()
        if not msg.data and self.robot_hablando:
            self.robot_hablando = False
            if self.turno_ocupado.is_set():
                self.turno_ocupado.clear()
                self.get_logger().info('Respuesta terminada. Escuchando de nuevo.')
        self.ignore_echo = msg.data

    def publicar_respuesta(self, texto):
        msg_voz = String()
        msg_voz.data = texto.strip() or 'No pude generar una respuesta.'
        self.tts_pub.publish(msg_voz)

    def rag_response_callback(self, future):
        try:
            response = future.result()
            if response.access_granted:
                self.get_logger().info(f'Acceso validado: {response.answer}')
            else:
                self.get_logger().warn(f'Acceso denegado: {response.answer}')

            respuesta = response.answer.strip() or 'No pude generar una respuesta.'
            self.ultima_respuesta_robot = respuesta.lower()
            self.publicar_respuesta(respuesta)
        except Exception as e:
            self.get_logger().error(f'El servidor RAG falló: {e}')
            self.publicar_respuesta('No pude consultar mi memoria en este momento.')

    # ==========================================
    # CAPTURA + VAD (segmentación de turnos)
    # ==========================================
    def escuchando_eco(self):
        return self.ignore_echo or (time.time() - self.fin_eco) < self.echo_tail_s

    def abrir_microfono(self):
        pa = pyaudio.PyAudio()
        idx = None if self.device_index < 0 else self.device_index
        info = pa.get_default_input_device_info() if idx is None \
            else pa.get_device_info_by_index(idx)
        rate = int(info['defaultSampleRate'])
        chunk = int(round(FRAME * rate / SAMPLE_RATE))
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                         input_device_index=idx, frames_per_buffer=chunk)
        self.get_logger().info(f"Micrófono '{info['name']}' a {rate} Hz")
        return pa, stream, rate, chunk

    def listen_loop(self):
        pa, stream, rate, chunk = self.abrir_microfono()
        x_src = np.linspace(0, 1, chunk, endpoint=False)
        x_dst = np.linspace(0, 1, FRAME, endpoint=False)

        contexto_vad = deque([np.zeros(FRAME, np.float32)] * 7, maxlen=7)
        preroll = deque(maxlen=int(300 / FRAME_MS))
        turno, en_voz, silencio, activos = [], False, 0, 0
        t_inicio = 0.0
        frames_silencio_fin = int(self.min_silence_ms / FRAME_MS)
        frames_max = int(self.max_turn_s * 1000 / FRAME_MS)

        self.get_logger().info('Micrófono listo. Puedes hablar...')
        try:
            while rclpy.ok():
                raw = stream.read(chunk, exception_on_overflow=False)
                if self.turno_ocupado.is_set():
                    contexto_vad.clear()
                    contexto_vad.extend([np.zeros(FRAME, np.float32)] * 7)
                    preroll.clear()
                    turno, en_voz, silencio, activos = [], False, 0, 0
                    continue

                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if chunk != FRAME:
                    x = np.interp(x_dst, x_src, x).astype(np.float32)

                if self.escuchando_eco():
                    # el robot está hablando: se descarta el audio y se reinicia el turno
                    turno, en_voz, silencio, activos = [], False, 0, 0
                    preroll.clear()
                    continue

                # Silero con ~250 ms de contexto; se toma la prob. de la última trama
                contexto_vad.append(x)
                prob = float(self.vad(np.concatenate(contexto_vad))[-1])

                if not en_voz:
                    preroll.append(x)
                    activos = activos + 1 if prob >= self.vad_on else 0
                    if activos >= 2:
                        en_voz, silencio = True, 0
                        turno = list(preroll)
                        t_inicio = time.time() - len(turno) * FRAME_MS / 1000
                        preroll.clear()
                    continue

                turno.append(x)
                silencio = silencio + 1 if prob < self.vad_off else 0

                if silencio >= frames_silencio_fin or len(turno) >= frames_max:
                    voz_frames = len(turno) - silencio
                    if voz_frames * FRAME_MS >= self.min_speech_ms:
                        audio = np.concatenate(turno[:voz_frames + 5])
                        self.turno_ocupado.set()
                        self.turnos.put((t_inicio, audio))
                    turno, en_voz, silencio, activos = [], False, 0, 0
        except Exception as e:
            self.get_logger().error(f'Error en el micrófono: {e}')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ==========================================
    # DIARIZACIÓN + ASOCIACIÓN CON ROSTROS
    # ==========================================
    def asignar_hablante(self, audio):
        h = huella_voz(audio)
        if h is None:
            return 'S1' if self.centroides else self.nuevo_hablante(None)
        if not self.centroides:
            return self.nuevo_hablante(h)
        sims = [float(np.dot(h, c)) for c in self.centroides]
        mejor = int(np.argmax(sims))
        self.get_logger().info(f'Similitud de voz con hablantes: {np.round(sims, 3).tolist()}')
        if sims[mejor] < self.speaker_threshold and len(self.centroides) < self.max_speakers:
            return self.nuevo_hablante(h)
        c = 0.8 * self.centroides[mejor] + 0.2 * h
        self.centroides[mejor] = c / np.linalg.norm(c)
        return f'S{mejor + 1}'

    def nuevo_hablante(self, h):
        self.centroides.append(h if h is not None else np.zeros(len(MEL_FB), np.float32))
        return f'S{len(self.centroides)}'

    def rostro_de_hablante(self, speaker, t_inicio):
        """Asocia el hablante con un rostro visto durante el turno.

        Cuenta cualquier rostro visto desde que empezó a hablar (menos face_timeout_s):
        la transcripción tarda varios segundos y "visible en el último segundo" fallaba.

        Si el hablante ya tenía un rostro y sigue visible, se conserva. Si no, se
        elige el rostro más grande (más cercano) que no pertenezca a otro hablante.
        """
        ahora = time.time()
        with self.lock:
            self.rostros = {tid: r for tid, r in self.rostros.items() if ahora - r['t'] < 30.0}
            visibles = {tid: r for tid, r in self.rostros.items()
                        if r['t'] >= t_inicio - self.face_timeout_s}
            if not visibles:
                return None, 'Desconocido'

            tid = self.speaker_a_track.get(speaker)
            if tid not in visibles:
                ocupados = {t for s, t in self.speaker_a_track.items()
                            if s != speaker and t in visibles}
                libres = [t for t in visibles if t not in ocupados] or list(visibles)
                tid = max(libres, key=lambda t: visibles[t]['area'])
                self.speaker_a_track[speaker] = tid
            return tid, visibles[tid]['name']

    # ==========================================
    # TRANSCRIPCIÓN (un turno a la vez)
    # ==========================================
    def transcribe_loop(self):
        while rclpy.ok():
            try:
                t_inicio, audio = self.turnos.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.procesar_turno(t_inicio, audio)
            except Exception as e:
                self.get_logger().error(f'Error procesando audio: {e}')
                self.turno_ocupado.clear()

    def procesar_turno(self, t_inicio, audio):
        duracion = len(audio) / SAMPLE_RATE
        self.get_logger().info(f'transcripción de turno de {duracion:.1f}s...')

        segments, _ = self.model.transcribe(
            audio, language='es', beam_size=1, condition_on_previous_text=False)
        texto = ' '.join(seg.text for seg in segments).strip()
        if not texto or len(texto) <= 5:
            self.turno_ocupado.clear()
            return

        texto_limpio = texto.lower()
        if self.ultima_respuesta_robot and (
                texto_limpio in self.ultima_respuesta_robot
                or self.ultima_respuesta_robot[:15] in texto_limpio):
            self.get_logger().warn(f'Se elimina el eco {texto}')
            self.turno_ocupado.clear()
            return

        speaker = self.asignar_hablante(audio)
        track_id, identidad = self.rostro_de_hablante(speaker, t_inicio)

        self.get_logger().info(
            f'[{speaker} | rostro {track_id} | {identidad}]: "{texto}"')

        msg = String()
        msg.data = json.dumps({
            'speaker': speaker,
            'track_id': track_id if track_id is not None else -1,
            'name': identidad,
            'text': texto,
            'start': t_inicio,
            'duration': round(duracion, 2),
        }, ensure_ascii=False)
        self.transcripcion_pub.publish(msg)

        # request para rag
        if not self.rag_client.service_is_ready():
            self.get_logger().warn('Servicio consultar_rag no disponible todavía.')
            self.publicar_respuesta('No pude conectarme a mi memoria en este momento.')
            return
        req = RagQuery.Request()
        req.identity = identidad
        req.question = texto
        future = self.rag_client.call_async(req)
        future.add_done_callback(self.rag_response_callback)


def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
