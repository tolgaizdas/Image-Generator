import pygame
import numpy as np


class ImagePlotter:
    def __init__(self, w, h, plot_name='Image Plotter', background_color='white'):
        self.width, self.height = w, h

        self.plot_name = plot_name
        self.background_color = background_color

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill(self.background_color)
        
        pygame.display.set_caption(self.plot_name)


    def _draw_line(self, angle, x, y, color, line_length, line_thickness):
        angle = angle - 90
        x_next = x + np.cos(np.radians(angle)) * line_length
        y_next = y + np.sin(np.radians(angle)) * line_length
        pygame.draw.line(self.screen, color, (x, y), (x_next, y_next), line_thickness)
        return x_next, y_next


    def draw_img(self, img, x, y, color='black', line_length=15, line_thickness=2):
        for angle in img:
            x, y = self._draw_line(angle, x, y, color, line_length, line_thickness)
    
    
    def display(self, delay=1000, keep_open=False):
        pygame.display.flip()
        pygame.time.wait(delay)
        self.screen.fill(self.background_color)
        if keep_open:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            pygame.quit()