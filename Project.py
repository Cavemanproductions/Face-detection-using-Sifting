import cv2
import os
                                        # folder address where all portraits are
portrait_dir = 'C:/Python/Portraits'

reference_images = []
reference_names = []
max_display_size = (800, 600)

for filename in os.listdir(portrait_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image file
        image = cv2.imread(os.path.join(portrait_dir, filename))
        if image is not None:
            if image.shape[0] > max_display_size[1] or image.shape[1] > max_display_size[0]:
                # Calculate the scaling factor
                scale = min(max_display_size[0] / image.shape[1], max_display_size[1] / image.shape[0])
                # Resize the image
                image = cv2.resize(image, None, fx=scale, fy=scale)
            reference_images.append(image)
            reference_names.append(os.path.splitext(filename)[0])  # Extract name from file name
for i in range(len(reference_images)):
    img_with_name = cv2.putText(reference_images[i], reference_names[i], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Reference Image", img_with_name)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Load query image (person to be identified)
query_image = cv2.imread('C:/Python/Raahim_portraits/dan.jpeg')
if query_image.shape[0] > max_display_size[1] or query_image.shape[1] > max_display_size[0]:
                scale = min(max_display_size[0] / query_image.shape[1], max_display_size[1] / query_image.shape[0])
                query_image = cv2.resize(query_image, None, fx=scale, fy=scale)
cv2.imshow("Reference Image", query_image)

# Create SIFT object
sift = cv2.SIFT_create()

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher()

# Detect keypoints and compute descriptors for reference images
reference_keypoints = []
reference_descriptors = []
for ref_image in reference_images:
    gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    reference_keypoints.append(keypoints)
    reference_descriptors.append(descriptors)

# Detect keypoints and compute descriptors for query image
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
query_keypoints, query_descriptors = sift.detectAndCompute(query_gray, None)

                    # Match keypoints between query image and reference images

best_match_count = 0
best_match_index = None

for i, ref_descriptors in enumerate(reference_descriptors):
    matches = flann.knnMatch(query_descriptors, ref_descriptors, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > best_match_count:
        best_match_count = len(good_matches)
        best_match_index = i

                                        # Identify the person

if best_match_index is not None:
    person_name = f"Person {best_match_index + 1}"
    print(f"The identified person is {person_name}")
else:
    print("No match found for the given person.")

                        # Results to be displayed. Side by side image matching

matched_image = cv2.drawMatches(query_image, query_keypoints, reference_images[best_match_index], reference_keypoints[best_match_index], good_matches, None)
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
