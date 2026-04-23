import cv2
import base64
import requests
import time

# After running `modal deploy modal_app.py`, replace this URL with the actual endpoint Modal provides.
# It typically takes the format: https://<username>--assistive-vision-backend-visionlanguagemodel-generate.modal.run
MODAL_ENDPOINT = "https://<your-username-here>--assistive-vision-backend-visionlanguagemodel-generate.modal.run"

def capture_and_encode_frame():
    """
    Captures a frame using OpenCV and encodes it as a Base64 JPEG string.
    Optimized for resource-constrained devices like Raspberry Pi.
    """
    # Initialize the camera stream (0 is typically the default Pi Camera or USB webcam)
    cap = cv2.VideoCapture(0)
    
    # Allow camera warmup if necessary
    time.sleep(1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")

    # Compress the image slightly to save bandwidth and reduce latency over Wi-Fi
    # Resize the image if needed: frame = cv2.resize(frame, (640, 480))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    
    if not result:
        raise RuntimeError("Failed to encode image to JPEG.")
        
    # Convert the bytes sequence to a Base64 encoded string
    b64_string = base64.b64encode(encimg).decode('utf-8')
    return b64_string

def request_vlm_inference(prompt: str, image_b64: str):
    """
    Constructs the JSON payload and blocks until the Modal endpoint responds.
    """
    # Structuring the payload matching the Pydantic InferencePayload schema in modal_app.py
    payload = {
        "prompt": prompt,
        "image_b64": image_b64
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending POST request to Mode endpoint: {MODAL_ENDPOINT}")
    start_time = time.time()
    
    try:
        response = requests.post(MODAL_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()  # Check for malformed requests / internal inference errors
        
        result = response.json()
        latency = (time.time() - start_time) * 1000
        print(f"Inference Latency: {latency:.2f} ms")
        print(f"VLM Response: {result.get('response')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Edge Communication Error: {e}")
        if response is not None:
             print("Server detail:", response.text)

if __name__ == "__main__":
    print("Assistive Vision Edge Client Initializing...")
    try:
        b64_image = capture_and_encode_frame()
        prompt = "Describe this scene out loud for a visually impaired user. What is directly in front of the camera?"
        request_vlm_inference(prompt, b64_image)
    except Exception as e:
        print(f"System Error: {str(e)}")
