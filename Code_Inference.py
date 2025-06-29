from ultralytics import YOLO
import cv2

# Path to save the trained model weights
MODEL_SAVE_PATH = r"C:\\Users\\asus\\Desktop\\Trained_Model\\Model_files\\runs\\detect\\plant_disease_detection_PLDD_2\\weights\\best.pt"

# Running Inference
def infer_yolov8(input_image_path):

    # Loading the trained model
    model = YOLO(MODEL_SAVE_PATH)

    # Performing inference on the input image
    results = model(input_image_path)

    # Extracting and print results
    for result in results:
        print("Detection Results:")
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Class ID
            label = model.names[cls_id]  # Class name
            confidence = box.conf[0].item()  # Confidence score
            print(f"Label: {label}, Confidence: {confidence:.2f}")

        # Visualizing the results (optional)
        annotated_frame = results[0].plot()
        cv2.imshow("Detections", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Performing inference
if __name__ == "__main__":
    INPUT_IMAGE_PATH = r"C:\\Users\\asus\\Desktop\\1afcd540-c746-4c02-8611-686ec35caaf5___GCREC_Bact.Sp 5654.JPG"
    infer_yolov8(INPUT_IMAGE_PATH)