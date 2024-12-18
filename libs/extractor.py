import cv2
import numpy as np
import pytesseract
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    index: int
    text: str
    bbox: Tuple[int, int, int, int]
    element_type: str

def sort_boxes_spatially(boxes):
    if not boxes:
        return []

    avg_height = np.mean([box[3] - box[1] for box in boxes])

    rows = []
    current_row = [boxes[0]]
    for box in boxes[1:]:
        if abs(box[1] - current_row[0][1]) < avg_height / 2:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    rows.sort(key=lambda row: row[0][1])

    if len(rows) > 1:
        avg_row_width = np.mean([row[-1][2] - row[0][0] for row in rows])
        page_width = max(box[2] for box in boxes)
        
        if avg_row_width < page_width * 0.7:
            columns = [[] for _ in range(int(page_width / avg_row_width) + 1)]
            for row in rows:
                for box in row:
                    col_index = int(box[0] / (page_width / len(columns)))
                    columns[col_index].append(box)
            
            sorted_boxes = []
            for column in columns:
                sorted_boxes.extend(sorted(column, key=lambda b: b[1]))
        else:
            sorted_boxes = [box for row in rows for box in row]
    else:
        sorted_boxes = [box for row in rows for box in row]

    return sorted_boxes

def collapse_overlapping_boxes(boxes):
    if not boxes:
        return []

    def merge_boxes(box1, box2):
        return (
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        )

    def is_overlapping(box1, box2):
        return not (box1[2] <= box2[0] or
                    box1[0] >= box2[2] or
                    box1[3] <= box2[1] or
                    box1[1] >= box2[3])

    merged = True
    while merged:
        merged = False
        result = []
        while boxes:
            current = boxes.pop(0)
            for other in boxes[:]:
                if is_overlapping(current, other):
                    current = merge_boxes(current, other)
                    boxes.remove(other)
                    merged = True
            result.append(current)
        boxes = result

    return boxes

def classify_text_element(text: str, prev_element_type: str) -> str:
    if len(text) < 100:
        if len(text) < 1:
            return "none"
        if prev_element_type == "heading":
            return "subheading"
        return "heading"
    return "paragraph"

def get_paragraph_bounding_boxes(image, initial_kernel_size=(5,5), max_paragraphs=25):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        kernel_sizes = [(5,5), (8,8), (10,10), (12,12)]  # Add more sizes if needed
        kernel_index = kernel_sizes.index(initial_kernel_size)

        while kernel_index < len(kernel_sizes):
            kernel = np.ones(kernel_sizes[kernel_index], np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = 1000
            paragraph_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area > min_area:
                    paragraph_boxes.append((x, y, x+w, y+h))
            
            collapsed_boxes = collapse_overlapping_boxes(paragraph_boxes)
            sorted_boxes = sort_boxes_spatially(collapsed_boxes)
            
            if len(sorted_boxes) <= max_paragraphs:
                logger.info(f"Total paragraphs detected: {len(sorted_boxes)} with kernel size {kernel_sizes[kernel_index]}")
                return sorted_boxes
            
            kernel_index += 1
        
        logger.warning(f"Unable to reduce paragraphs below {max_paragraphs}. Using largest kernel size.")
        return sorted_boxes

    except Exception as e:
        logger.exception(f"An error occurred in get_paragraph_bounding_boxes: {str(e)}")
        return []

def extract_text_from_image(image):
    try:
        text_elements = []
        paragraph_boxes = get_paragraph_bounding_boxes(image)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prev_element_type = ""

        for index, (x1, y1, x2, y2) in enumerate(paragraph_boxes, start=1):
            paragraph = pil_image.crop((x1, y1, x2, y2))
            text = pytesseract.image_to_string(paragraph).strip()
            
            element_type = classify_text_element(text, prev_element_type)

            if element_type not in ["header", "footer"]:
                text_elements.append(TextElement(index, text, (x1, y1, x2, y2), element_type))
                prev_element_type = element_type

        return text_elements
    except Exception as e:
        logger.exception(f"An error occurred in extract_text_from_image: {str(e)}")
        return []