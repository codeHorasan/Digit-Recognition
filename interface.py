import pygame
from keras.models import load_model
from tkinter import *
from tkinter import messagebox
from PIL import Image
import numpy as np
import tensorflow as tf
pygame.init()

screen = pygame.display.set_mode((400,400))
pygame.display.set_caption("Number Prediction")

run = True
while run:
    for event in pygame.event.get():
        if pygame.mouse.get_pressed()[0]:
            pygame.draw.rect(screen, (255, 255, 255), (event.pos[0], event.pos[1], 50, 50))
            pygame.display.update()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            my_model = load_model('model.h5')
            pygame.image.save(screen, "image.jpg")

            image = Image.open("image.jpg")
            image = image.resize((28,28))
            image = image.convert('L')
            image = np.array(image)
            image = image.reshape(1, 28, 28)
            image = tf.keras.utils.normalize(image, axis=1)

            pred = my_model.predict([image])[0]

            window = Tk()
            window.withdraw()
            info_string = "Predicted number is {}\nPossibility is {}".format(np.argmax(pred), max(pred) * 100)
            messagebox.showinfo("Prediction", info_string)
            window.destroy()

        if event.type == pygame.KEYUP and event.key == pygame.K_c:
            pygame.draw.rect(screen, (0,0,0), (0,0,400,400))
            pygame.display.update()

        if event.type == pygame.QUIT:
            run = False

pygame.quit()
