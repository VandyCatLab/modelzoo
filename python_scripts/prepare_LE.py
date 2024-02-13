from PIL import Image
import csv

def extract_images(image_path):
    # Load the main image
    main_image = Image.open(image_path)

    # Initialize the list to store extracted images
    extracted_images = []

    # Dimensions of each small image
    img_width = img_height = 136

    # Coordinates of the first image
    x, y = 0, 2

    # Extract each image
    for i in range(3):
        # Define the area to crop
        left = x + (i * (img_width + 96))  # 96px is the space between images
        upper = y
        right = left + img_width
        lower = upper + img_height

        # Crop and save the image
        cropped_img = main_image.crop((left, upper, right, lower))
        extracted_images.append(cropped_img)

    return extracted_images

csv_file = '../data_storage/learning_exemplar_trials.csv'

with open(csv_file, newline='') as c:
        csv_data = csv.reader(c)
        image_sets = []
        for idx, row in enumerate(csv_data):
            if idx > 0:
                images = extract_images(f'../data_storage/standalone/LearningExemplars/stimuli/{row[1]}')
                
                for idx, im in enumerate(images):
                    name = f'../LE_set/{row[1][:-4]}_{idx+1}.jpg'
                    im.save(name)
            