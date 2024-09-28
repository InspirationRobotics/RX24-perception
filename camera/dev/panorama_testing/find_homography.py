import cv2
import numpy as np
from pathlib import Path


def read_images(image_folder : Path, index : int):
    # Read images from the folder
    image1 = cv2.imread(str(image_folder / f"port_{index}.jpg"))
    image2 = cv2.imread(str(image_folder / f"starboard_{index}.jpg"))
    return image1, image2

def find_matches(image1, image2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors from both frames
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Use FLANN-based matcher to find matches between descriptors
    index_params = dict(algorithm=1, trees=5)  # For SIFT/ORB
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches for visualization (optional)
    image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)
    cv2.imshow("Matches", image_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return keypoints1, keypoints2, good_matches

def find_homography_matrix(keypoints1, keypoints2, good_matches):
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography

def stitch_images(image1, image2, homography):
    # Warp the second image using the homography matrix
    # Get the dimensions of the images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Warp the second image to align with the first one
    warped_image2 = cv2.warpPerspective(image2, homography, (w1 + w2, h1))  # (w1+w2, h1) is the output size

    # Create a composite image by placing the first image on the canvas
    composite_image = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
    composite_image[0:h1, 0:w1] = image1

    # Overlay the warped second image
    for i in range(w1):
        for j in range(h1):
            if warped_image2[j, i].any() != 0:  # If there is data from the second image
                composite_image[j, i] = warped_image2[j, i]  # Replace with the warped image

    return composite_image


def find_homography(path : Path | str, index : int):
    # Read images
    image_folder = Path(path)
    image1, image2 = read_images(image_folder, index)

    # Find matches
    keypoints1, keypoints2, good_matches = find_matches(image1, image2)

    # Find homography
    homography = find_homography_matrix(keypoints1, keypoints2, good_matches)

    # Stitch images
    composite_image = stitch_images(image1, image2, homography)

    cv2.imshow("Composite Image", composite_image)
    cv2.waitKey(0)

def main():
    image_folder = Path("sample_images")
    # for i in range(1, 6):
    find_homography(image_folder, 1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()