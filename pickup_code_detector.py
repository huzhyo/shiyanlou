# Main script for the pickup code detector
# This script uses OpenCV and Pytesseract to find and highlight pickup codes in images.
import cv2
import numpy as np
import pytesseract
from PIL import Image # Imported as Pytesseract might require it indirectly.
import argparse
import os # For path operations (e.g., creating directories, joining paths)

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path using OpenCV.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array in BGR format (OpenCV default).

    Raises:
        FileNotFoundError: If the image cannot be found at the given path or
                           if the file cannot be opened/read by OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        # This check handles cases where the file doesn't exist or is not a valid image format.
        raise FileNotFoundError(f"Image not found or could not be loaded from path: {image_path}")
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the input image to enhance text visibility for OCR.
    The steps include:
    1. Convert to grayscale: Simplifies the image, reducing complexity for subsequent processing.
    2. Apply Gaussian blur: Smooths the image to reduce noise that can interfere with OCR.
    3. Apply adaptive thresholding: Binarizes the image (converts to black and white)
       to clearly separate text from the background, especially under non-uniform
       lighting conditions.

    Args:
        image (np.ndarray): The input image in OpenCV BGR format.

    Returns:
        np.ndarray: The preprocessed (binary) image, suitable for text detection.
    """
    # Convert the image from BGR (OpenCV default) to grayscale.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce high-frequency noise.
    # A 5x5 kernel is a common choice; its size can be tuned depending on image characteristics.
    # The '0' for sigmaX means OpenCV calculates it from the kernel size.
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding. This method is better than simple global thresholding
    # for images with varying illumination.
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Threshold value is a Gaussian-weighted sum of
    #                                 neighborhood values.
    # cv2.THRESH_BINARY_INV: Inverts the thresholding result. If text is darker than
    #                        the background, it becomes white (foreground) and background black.
    #                        This is often preferred for contour detection and Tesseract.
    # 11: Block size (size of the pixel neighborhood used to calculate the threshold).
    # 2: Constant C subtracted from the mean or weighted sum. It's a fine-tuning parameter.
    threshold_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return threshold_image

def detect_text_regions(preprocessed_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detects potential text regions in a preprocessed (binary) image using contour detection.
    It filters these contours based on heuristic measures of size and aspect ratio
    to identify rectangular areas that are likely to contain text.

    Args:
        preprocessed_image (np.ndarray): The binary image (output from preprocess_image).

    Returns:
        list[tuple[int, int, int, int]]: A list of tuples, where each tuple represents
                                          a bounding box (x, y, width, height) of a
                                          detected potential text region.
    """
    # Find contours in the preprocessed image.
    # cv2.RETR_EXTERNAL: Retrieves only the extreme outer contours. This is useful for
    #                    finding distinct text blocks.
    # cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments
    #                          and leaves only their end points, saving memory and computation.
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions_bounding_boxes = []
    for contour in contours:
        # Get the bounding rectangle (x, y, width, height) for each contour.
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic filtering criteria (these are heuristic and may need adjustment based on image characteristics)
        # Filter by width and height to ignore very small noise or very large non-text areas.
        min_width, max_width = 10, 500   # Example: minimum 10 pixels wide, maximum 500
        min_height, max_height = 10, 100 # Example: minimum 10 pixels high, maximum 100
        
        if min_width < w < max_width and min_height < h < max_height:
            # Calculate aspect ratio (width / height).
            aspect_ratio = w / float(h)
            # Filter by aspect ratio. Text characters and words usually fall within a certain range.
            # This helps filter out shapes that are too thin or too square/tall to be typical text.
            min_aspect_ratio, max_aspect_ratio = 0.5, 5.0 # Example: aspect ratio between 0.5 and 5.0
            
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                text_regions_bounding_boxes.append((x, y, w, h))
                
    return text_regions_bounding_boxes

def perform_ocr_on_regions(
    image: np.ndarray, 
    text_regions_bounding_boxes: list[tuple[int, int, int, int]]
) -> list[dict]:
    """
    Performs Optical Character Recognition (OCR) on specified regions of an image.
    It iterates through each provided bounding box, crops that region from the
    image, and uses Pytesseract to extract text.

    Args:
        image (np.ndarray): The original, unprocessed image (OpenCV BGR format).
                            Regions will be cropped from a grayscale version of this image.
        text_regions_bounding_boxes (list[tuple[int, int, int, int]]): A list of
            bounding boxes (x, y, w, h) defining the regions for OCR.

    Returns:
        list[dict]: A list of dictionaries. Each dictionary corresponds to a region
            where text was successfully extracted and contains:
            - 'text' (str): The extracted text, stripped of leading/trailing whitespace.
            - 'bbox' (tuple[int, int, int, int]): The bounding box (x, y, w, h)
                                                  of the region from which the text was extracted.
            Returns an empty list if no text is found in any of the regions or if
            text_regions_bounding_boxes is empty.
    """
    ocr_results = []
    # Convert the original image to grayscale. Tesseract often performs better with grayscale images,
    # and it simplifies the input by removing color information.
    gray_image_for_ocr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in text_regions_bounding_boxes:
        # Crop the region of interest (ROI) from the grayscale image using numpy slicing.
        cropped_region = gray_image_for_ocr[y:y+h, x:x+w]
        
        try:
            # Before attempting OCR, check if the cropped region is valid (has non-zero size).
            # An empty region can cause errors in Pytesseract.
            if cropped_region.size == 0:
                # This might happen if the bounding box coordinates are outside the image
                # or if width/height is zero.
                print(f"Warning: Cropped region at {(x,y,w,h)} is empty. Skipping OCR for this region.")
                continue

            # Use Pytesseract to extract text from the cropped image region.
            # '--psm 6': Assumes a single uniform block of text. This Page Segmentation Mode (PSM)
            #            can be crucial for accuracy. Other modes (0-13) can be experimented with.
            #            For example, PSM 7 treats the image as a single text line.
            extracted_text = pytesseract.image_to_string(cropped_region, config='--psm 6')
            extracted_text = extracted_text.strip() # Remove leading/trailing whitespace and newlines.
            
            # Only add to results if Tesseract found some text.
            if extracted_text: 
                ocr_results.append({'text': extracted_text, 'bbox': (x, y, w, h)})
        except pytesseract.TesseractError as e:
            # Handle errors specific to Tesseract OCR (e.g., Tesseract not installed correctly,
            # language data missing, or an issue with processing the image region).
            print(f"Tesseract OCR Error on region {(x,y,w,h)}: {e}")
        except Exception as e:
            # Handle any other unexpected errors that might occur during OCR for a specific region.
            print(f"Unexpected error during OCR on region {(x,y,w,h)}: {e}")
            
    return ocr_results

def search_and_highlight_code(
    image: np.ndarray, 
    ocr_results: list[dict], 
    target_code: str, 
    highlight_color: tuple[int, int, int] = (0, 0, 255), # Default BGR: Red
    thickness: int = 2
) -> tuple[np.ndarray, bool]:
    """
    Searches for a specific `target_code` string within the text extracted by OCR
    from various image regions. If found, it highlights the corresponding bounding
    box(es) on a copy of the original image.

    Args:
        image (np.ndarray): The original image (OpenCV BGR format) on which to draw highlights.
        ocr_results (list[dict]): A list of dictionaries, where each item should have
                                  a 'text' field (str) and a 'bbox' field 
                                  (tuple[int, int, int, int]).
        target_code (str): The specific code string to search for. The search is
                           performed case-insensitively.
        highlight_color (tuple[int, int, int], optional): The BGR color tuple for the
                                                           highlighting rectangle.
                                                           Defaults to (0, 0, 255) (red).
        thickness (int, optional): The thickness of the rectangle's border line.
                                   Defaults to 2.

    Returns:
        tuple[np.ndarray, bool]: A tuple containing:
            - output_image (np.ndarray): A copy of the original image with the
                                         target code(s) highlighted. If the code
                                         is not found, this is an unmodified copy.
            - found (bool): True if the target code was found in any of the OCR'd text
                            regions, False otherwise.
    """
    output_image = image.copy() # Work on a copy to avoid modifying the original image.
    found = False
    for item in ocr_results:
        extracted_text = item['text']
        x, y, w, h = item['bbox'] # Unpack the bounding box coordinates and dimensions.
        
        # Perform a case-insensitive search for the target_code within the extracted_text.
        # This makes the search more robust (e.g., "pickup123" will match "Pickup123").
        if target_code.lower() in extracted_text.lower():
            found = True
            # Draw a rectangle on the `output_image` to highlight the found code.
            # (x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner.
            cv2.rectangle(output_image, (x, y), (x + w, y + h), highlight_color, thickness)
            # Note: This implementation continues searching and highlights all occurrences.
            # If only the first occurrence should be highlighted, a 'break' statement could be added here.
            
    return output_image, found

# The original main() function with hardcoded example usage is removed 
# in favor of main_cli() which accepts command-line arguments.
# If you need to test with hardcoded paths, you can temporarily reinstate it
# or, preferably, use the command line as intended.

# def main(): # This is the old main function, kept here for reference if needed.
#     print("Pickup code detector script - Old Main Function")
#     # Example usage (for testing, can be commented out later):
#     # import os
#     # if not os.path.exists("images"):
#     #     os.makedirs("images")
#     # sample_image_path = "images/sample.jpg" # Replace with a real image path for testing
#     # # Create a dummy sample.jpg for testing if it doesn't exist
#     # # if not os.path.exists(sample_image_path) and os.path.exists("images"):
#     # #     dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
#     # #     cv2.imwrite(sample_image_path, dummy_image)
#     #
#     # try:
#     #     original_image = load_image(sample_image_path)
#     #     preprocessed_image = preprocess_image(original_image)
#     #     text_regions = detect_text_regions(preprocessed_image)
#     #
#     #     image_with_boxes = original_image.copy()
#     #     for region in text_regions:
#     #         x, y, w, h = region
#     #         cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     #
#     #     ocr_data = perform_ocr_on_regions(original_image, text_regions)
#     #     print("\nOCR Results:")
#     #     if ocr_data:
#     #         for item in ocr_data:
#     #             print(f"  Text: '{item['text']}', BBox: {item['bbox']}")
#     #     else:
#     #         print("  No text found by OCR.")
#     #
#     #     sample_target_code = "PICKUP123" 
#     #     highlighted_image, found_code = search_and_highlight_code(original_image, ocr_data, sample_target_code)
#     #
#     #     if found_code:
#     #         print(f"\nPickup code '{sample_target_code}' found.")
#     #         if not os.path.exists("images"):
#     #             os.makedirs("images")
#     #         cv2.imwrite("images/highlighted_output.jpg", highlighted_image)
#     #         print("Highlighted image saved to images/highlighted_output.jpg")
#     #         # cv2.imshow("Highlighted Code", highlighted_image) # Local debugging
#     #         # cv2.waitKey(0) # Local debugging
#     #     else:
#     #         print(f"\nPickup code '{sample_target_code}' not found.")
#     #
#     #     # cv2.imshow("Preprocessed Image", preprocessed_image) # Local debugging
#     #     # cv2.imshow("Detected Text Regions", image_with_boxes) # Local debugging
#     #     # cv2.waitKey(0) # Local debugging
#     #     # cv2.destroyAllWindows() # Local debugging - call once if any imshow is used
#     # except FileNotFoundError as e:
#     #     print(e)
#     # except Exception as e: 
#     #     print(f"An error occurred in old main: {e}")


def main_cli():
    """
    Main command-line interface (CLI) function for the pickup code detector.

    This function orchestrates the entire detection process:
    1.  Parses command-line arguments for the image path and the target pickup code.
    2.  Loads the specified image.
    3.  Preprocesses the image to enhance text for OCR.
    4.  Detects potential text regions within the preprocessed image.
    5.  Performs OCR on these detected regions to extract text.
    6.  Searches for the user-specified target code among the OCR results.
    7.  If the code is found, it's highlighted on the original image, and the
        result is saved to `images/highlighted_output.jpg`.
    8.  Prints progress, results, and error messages to the console.

    The function includes error handling for common issues like file not found or
    Tesseract OCR errors.
    """
    # Setup argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Detect and highlight specific pickup codes in an image using OpenCV and Tesseract OCR."
    )
    parser.add_argument(
        "image_path", 
        help="Path to the input image file (e.g., 'images/my_package.jpg')."
    )
    parser.add_argument(
        "target_code", 
        help="The specific pickup code string to search for (e.g., 'PKUP123XYZ')."
    )
    args = parser.parse_args()

    try:
        # Step 1: Load Image
        print(f"Attempting to load image from: {args.image_path}")
        original_image = load_image(args.image_path)
        print(f"Image '{args.image_path}' loaded successfully.")

        # Step 2: Preprocess Image
        print("Preprocessing image...")
        preprocessed_img = preprocess_image(original_image)
        print("Image preprocessed successfully.")
        # For local debugging, you might want to save or display the preprocessed image:
        # cv2.imwrite("images/debug_preprocessed.jpg", preprocessed_img)
        # cv2.imshow("Preprocessed Image - Debug", preprocessed_img); cv2.waitKey(0) # Debug: Show preprocessed

        # Step 3: Detect Text Regions
        print("Detecting potential text regions...")
        text_regions = detect_text_regions(preprocessed_img)
        print(f"Detected {len(text_regions)} potential text regions.")
        if not text_regions:
            print("No potential text regions were detected. Cannot proceed with OCR. Exiting.")
            return # Exit gracefully if no regions are found

        # Step 4: Perform OCR on Detected Regions
        print("Performing OCR on detected regions...")
        ocr_results = perform_ocr_on_regions(original_image, text_regions) 
        print(f"OCR completed. {len(ocr_results)} regions yielded recognizable text.")
        if not ocr_results:
            print("OCR did not yield any usable text from the detected regions. Exiting.")
            return # Exit if OCR results are empty

        print("\n--- Extracted OCR Data ---")
        for i, item in enumerate(ocr_results):
            # Print each piece of extracted text and its bounding box
            print(f"  Region {i+1}: BBox: {item['bbox']}, Text: '{item['text']}'")
        print("--------------------------")

        # Step 5: Search for the Target Code and Highlight
        print(f"\nSearching for target code: '{args.target_code}' in OCR results...")
        output_dir = "images" # Define the directory to save output images
        # Ensure the output directory exists, create it if it doesn't.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")
        # Define the full path for the output image file
        output_image_path = os.path.join(output_dir, "highlighted_output.jpg")
        
        highlighted_image, found = search_and_highlight_code(
            original_image, ocr_results, args.target_code
        )

        if found:
            print(f"\nSUCCESS: Target pickup code '{args.target_code}' was FOUND in the image.")
            # Save the image with highlighted codes
            cv2.imwrite(output_image_path, highlighted_image)
            print(f"Highlighted image has been saved to: '{output_image_path}'")
            
            # The cv2.imshow lines below are for local debugging and require a desktop environment.
            # They are commented out to ensure script compatibility in headless environments.
            # print("Displaying highlighted image (press any key to close window)...")
            # cv2.imshow("Highlighted Pickup Code", highlighted_image)
            # cv2.waitKey(0) # Wait for a key press before closing the window
        else:
            print(f"\nINFO: Target pickup code '{args.target_code}' was NOT FOUND in the image.")

    except FileNotFoundError as e:
        # Handle file not found errors specifically for the input image.
        print(f"Error: Input image file not found. Details: {e}")
    except pytesseract.TesseractError as e:
        # Handle errors originating from Tesseract OCR engine.
        print(f"Tesseract OCR Error: {e}")
        print("Ensure Tesseract OCR is correctly installed and configured on your system.")
        print("This might involve adding Tesseract to your system's PATH or setting the TESSDATA_PREFIX environment variable.")
    except cv2.error as e:
        # Handle errors originating from OpenCV operations.
        print(f"OpenCV Error: {e}")
        print("This could be due to an issue with the image file or an OpenCV internal problem.")
    except Exception as e:
        # Catch any other unexpected errors during the process.
        print(f"An unexpected error occurred: {e}")
    
    # finally:
        # If cv2.imshow() was used for debugging, cv2.destroyAllWindows() should be called here
        # to close any OpenCV display windows. Since they are commented out, this is also commented.
        # cv2.destroyAllWindows() # Important for closing OpenCV windows if cv2.imshow is used.

if __name__ == "__main__":
    # This check ensures that main_cli() is called only when the script is executed
    # directly (e.g., `python pickup_code_detector.py ...`), not when it's imported
    # as a module into another script.
    main_cli()
