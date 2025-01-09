import cv2
import pytesseract

class LicensePlateProcessor:
    def __init__(self, tesseract_cmd=None):
        # Đặt đường dẫn đến tesseract nếu cần
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def crop_license_plate(self, frame, bbox):
        """
        Cắt hình ảnh biển số xe từ khung hình dựa trên tọa độ bbox.
        Args:
            frame: Ảnh gốc (numpy array).
            bbox: Tọa độ hình chữ nhật của biển số xe (x1, y1, x2, y2).

        Returns:
            Cropped image: Ảnh biển số xe đã được cắt.
        """
        x1, y1, x2, y2 = bbox
        cropped_image = frame[y1:y2, x1:x2]
        return cropped_image

    def extract_text_from_plate(self, plate_image):
        """
        Sử dụng Tesseract OCR để nhận dạng ký tự từ ảnh biển số xe.
        Args:
            plate_image: Ảnh biển số xe đã cắt.

        Returns:
            text: Ký tự nhận dạng được từ biển số.
        """
        # Chuyển đổi ảnh sang grayscale để tăng độ chính xác
        gray_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # Áp dụng threshold để làm nổi bật ký tự
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Nhận diện ký tự bằng Tesseract
        text = pytesseract.image_to_string(thresh_image, config='--psm 8')
        return text.strip()
