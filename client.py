import asyncio
import websockets
import json

async def send_audio():
    uri = "ws://127.0.0.1:8000/ws/analyze"
    file_path = "test-" + input("Choose file: ") + ".ogg"

    async with websockets.connect(uri) as websocket:
        print(f"Serverga ulandi. Fayl yuborilmoqda: {file_path}...")
        
        with open(file_path, "rb") as f:
            audio_data = f.read()
            await websocket.send(audio_data)

        print("Fayl yuborildi. Javob kutilmoqda...\n")
        print("-" * 50)

        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if "status" in data:
                    print(f"[STATUS]: {data['message']}")
                
                elif data.get("type") == "segment":
                    start = data['start_ms'] / 1000
                    end = data['end_ms'] / 1000

		    # 🔥 O'ZGARISH SHU YERDA: Ism va ishonch foizini to'g'ridan-to'g'ri chiqaramiz
                    spk_id = data.get('speaker_id', '')
                    spk_name = data.get('speaker_name', 'Unknown')
                    confidence = data.get('confidence', 0.0)

                    text = data['text']
                    print(f"[{start:.1f}s - {end:.1f}s] {spk_id} (Ism: {spk_name}, Ishonch: {confidence:.2f}): {text}")

                
                elif data.get("type") == "finished":
                    print("-" * 50)
                    print(f"[TUGADI] Jami vaqt: {data['total_time']}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("Server aloqani uzdi.")

if __name__ == "__main__":
    asyncio.run(send_audio())
