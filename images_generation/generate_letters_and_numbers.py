# -*- coding: utf-8 -*-

# -------------------------------- Imports ------------------------------#

# Import python imaging libs
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import glob
import os

# Import operating system lib
import os

# Import random generator
from random import randint


# -------------------------------- Cleanup ------------------------------#

def Cleanup():
    # Delete ds_store file
    if os.path.isfile(font_dir + '.DS_Store'):
        os.unlink(font_dir + '.DS_Store')

        # Delete all files from output directory
    for file in os.listdir(out_dir):
        file_path = os.path.join(out_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    return


# --------------------------- Generate Characters -----------------------#

def GenerateCharacters(characters, type):
    # Counter
    k = 1
    # Process the font files
    for dirname, dirnames, filenames in os.walk(font_dir):
        # For each font do
        for fontname in filenames:
            # Get font full file path
            font_resource_file = os.path.join(dirname, fontname)

            # For each character do
            for char in characters:
                # For each font size do
                for font_size in font_sizes:
                    if font_size > 0:
                        # For each background color do
                        for background_color in background_colors:

                            # NOT NECESSARY FOR PYTHON 3
                            # Convert the character into unicode
                            # character = str(char, 'utf-8')

                            character = char

                            # Specify font : Resource file, font size
                            font = ImageFont.truetype(font_resource_file, font_size)

                            # Get character width and height
                            (font_width, font_height) = font.getsize(character)

                            padding = 10

                            # Create character image :
                            # Grayscale, image size, background color
                            char_image = Image.new('L', (image_size,
                                                         image_size), background_color)

                            # Draw character image
                            draw = ImageDraw.Draw(char_image)

                            # Calculate x position
                            x = (image_size - font_width) / 2

                            # Calculate y position
                            y = (image_size - font_height) / 2

                            # Draw text : Position, String,
                            # Options = Fill color, Font
                            draw.text((x, y), character, (245 -
                                                          background_color) + randint(0, 10), font=font)

                            '''
                            Naming would be the following - character + font + background
                            Then when reading, read just the first character of the filename and that will be the label
                            Technically for vehicle VINS I need only capital letters but for now I will leave lowercase as well
                            '''
                            file_name = ''
                            if type == 'letters':
                                # Final file name
                                file_name = letters_dir + '/' + character + '_' + fontname + '_' + str(font_size) + '_' + str(background_color) + '.png'

                            elif type == 'numbers':
                                # Final file name
                                file_name = numbers_dir + '/' + character + '_' + fontname + '_' + str(font_size) + '_' + str(background_color) + '.png'

                            elif type == 'capital_letters':
                                # Final file name
                                file_name = capital_letters_dir + '/' + character + '_' + fontname + '_' + str(font_size) + '_' + str(background_color) + '.png'

                            # Save image
                            char_image.save(file_name)

                            # Increment counter
                            k = k + 1
    return


# --------------------------- Generate VIN number  -----------------------#

def GenerateVIN(n):
    '''
    VIN CONTAINS 17 numbers
    letters are capital
    1 number
    4 letters
    5 numbers
    1 letter
    6 numbers

     I (i), O (o), Q (q), U (u) or Z (z), or the number 0  are not allowed in VIN numbers
    '''
    # Process the font files


    for _ in range(n):

        # draw the vin
        first_number = ''.join(random.choice(numbers)) # ''.join in order to convert lists to strings
        four_letters = ''.join(random.sample(capital_letters, 4))
        five_numbers = ''.join(random.sample(numbers, 5))
        one_letter = ''.join(random.choice(capital_letters))
        six_numbers = ''.join(random.sample(numbers, 6))

        for dirname, dirnames, filenames in os.walk(font_dir):
            # For each font do
            for fontname in filenames:
                # Get font full file path
                font_resource_file = os.path.join(dirname, fontname)

                # For each font size do
                for font_size in font_sizes:
                    if font_size > 0:
                        # For each background color do
                        for background_color in background_colors:
                            # Specify font : Resource file, font size
                            font = ImageFont.truetype(font_resource_file, font_size)

                            vin_number_text = str(first_number) + four_letters + str(five_numbers) + one_letter + str(six_numbers)

                            # get the line size
                            text_width, text_height = font.getsize(vin_number_text)

                            width = vin_width  # 17 characters + 1 margin on both sides


                            #canvas = Image.new('L', (image_size, vin_width), background_color)
                            canvas = Image.new('L', (width, vin_height), background_color)

                            # Calculate x position
                            x = (width - text_width) / 2

                            # Calculate y position
                            y = (vin_height - text_height) / 2

                            # draw the text onto the text canvas, and use black as the text color
                            draw = ImageDraw.Draw(canvas)
                            draw.text((x, y), vin_number_text, (245 - background_color) + randint(0, 10), font=font)

                            '''
                            Filename is the label for that VIN number
                            When reading just separate first 17 characters or everything before underscore and that will be the label
                            '''

                            # Final file name
                            file_name = vins_dir + '/' + vin_number_text + '_' + fontname + str(background_color) + '.png'

                            # Save image
                            canvas.save(file_name)

    return

# ------------------------------- Input and Output ------------------------#

# Directory containing fonts
font_dir = 'fonts'

letters_dir = 'output/letters'

capital_letters_dir = '../../DR_data/capital_letters' #'output/capital_letters'

numbers_dir = '../../DR_data/numbers' #'output/numbers'

vins_dir = 'output/vins'

# Output
out_dir = 'output'

# ---------------------------------- Characters ---------------------------#

# Numbers
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# Small letters
small_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'y']

# Capital letters
capital_letters = [letter.upper() for letter in small_letters]

# Select characters
#characters = numbers + small_letters + capital_letters

# ---------------------------------- Colors -------------------------------#

# Background color
white_colors = (200, 205, 215, 225, 235, 245, 250)
black_colors = (0, 10, 20, 30, 40, 50)
gray_colors = (135, 145, 155)

background_colors = white_colors + black_colors + gray_colors

# ----------------------------------- Sizes -------------------------------#

# Character sizes
#small_sizes = (8, 12, 16)
#medium_sizes = (, 24, 28)
large_sizes = (28, 30, 32, 36, 40, 48)
larger_sizes = (50, 52, 54, 56, 58, 60, 62, 64, 68, 70, 72, 74)


font_sizes = large_sizes + larger_sizes

# Image size
image_size = 64

# For VIN
vin_height = int(image_size * 0.7)
vin_width = 600

def remove_files(path):
    files_cropped = glob.glob(path + '/*')
    for f in files_cropped:
        os.remove(f)

# ----------------------------------- Main --------------------------------#

# Do cleanup
Cleanup()

# Generate characters
#GenerateCharacters(numbers, 'numbers')
#GenerateCharacters(small_letters, 'letters')
GenerateCharacters(capital_letters, 'capital_letters')

#remove_files('output/vins')
#GenerateVIN(1000)