import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3

# If Tesseract is not in your PATH, specify the path to the executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)

def process_image(frame):
    """Process the frame to extract text using Tesseract OCR."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # OCR (Optical Character Recognition) on the image
    custom_config = r'--oem 3 --psm 6'  # Set Tesseract configurations
    text = pytesseract.image_to_string(thresh, config=custom_config)

    return text, thresh  # Return OCR result and the processed image

def read_document_continuously():
    """Continuously read frames from the camera, process them, and output detected text with voice."""
    # Open the webcam or a video stream
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    if not cap.isOpened():
        
        print("Error: Could not open video stream")
        return

    # Set up matplotlib figure for displaying camera feed
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    previous_text = ""  # Store the previous text to avoid repeated announcements

    print("Press 'q' to exit the loop.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Process the frame and extract text
        text, processed_frame = process_image(frame)

        # Check if the detected text has changed
        if text.strip() and text != previous_text:
            print("Detected text:", text)
            engine.say(text)  # Speak the detected text
            engine.runAndWait()  # Wait until speaking is complete
            previous_text = text  # Update the previous text

        # Use matplotlib to display the processed frame
        ax.clear()
        ax.imshow(processed_frame, cmap="gray")
        ax.set_title("Processed Frame (Press 'q' to Quit)")
        ax.axis("off")
        plt.draw()
        plt.pause(0.01)

        # # Check if 'q' is pressed to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Exiting...")
        #     break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    read_document_continuously()

# import cv2
# import pytesseract
# import numpy as np
# import matplotlib.pyplot as plt
# import pyttsx3

# # If Tesseract is not in your PATH, specify the path to the executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 125)

# def process_image(frame):
#     """Process the frame to extract text using Tesseract OCR."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Apply thresholding to get a binary image
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#     # OCR (Optical Character Recognition) on the image
#     custom_config = r'--oem 3 --psm 6'  # Set Tesseract configurations
#     text = pytesseract.image_to_string(thresh, config=custom_config)

#     return text.strip(), thresh  # Return OCR result and the processed image

# def read_document_once():
#     """Capture a frame, process it, and stop after reading the document aloud."""
#     # Open the webcam
#     cap = cv2.VideoCapture(1)  # Use 0 or the appropriate camera index

#     if not cap.isOpened():
#         print("Error: Could not open video stream")
#         return

#     # Set up matplotlib figure for displaying camera feed
#     plt.ion()
#     fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#     print("Show a document to the camera. The program will read it aloud and then stop.")

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Failed to capture image")
#             break

#         # Process the frame and extract text
#         text, processed_frame = process_image(frame)

#         # If meaningful text is detected, read it aloud and stop
#         if text:
#             print("Detected text:", text)
#             engine.say(text)  # Speak the detected text
#             engine.runAndWait()  # Wait until speaking is complete
#             break  # Exit the loop after reading

#         # Use matplotlib to display the processed frame
#         ax.clear()
#         ax.imshow(processed_frame, cmap="gray")
#         ax.set_title("Processed Frame (Press 'q' to Quit)")
#         ax.axis("off")
#         plt.draw()
#         plt.pause(0.01)

#     # Release the webcam and close any OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()
#     plt.ioff()  # Turn off interactive mode
#     plt.show()
#     print("Done reading the document.")

# if __name__ == "__main__":
#     read_document_once()

