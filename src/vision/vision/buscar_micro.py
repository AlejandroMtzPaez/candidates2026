import pyaudio

def main():
    p = pyaudio.PyAudio()
    print("\n" + "="*40)
    print("   LISTA DE MICRÓFONOS ENCONTRADOS   ")
    print("="*40)
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        # Solo mostrar los que pueden grabar audio (canales de entrada > 0)
        if dev.get('maxInputChannels') > 0:
            print(f"ID del Micrófono: {i}  |  Nombre: {dev.get('name')}")
            
    print("="*40 + "\n")
    p.terminate()

if __name__ == "__main__":
    main()

