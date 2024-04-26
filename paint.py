import pygame
import math


def calculate_angle(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    angle = math.degrees(math.atan2(dx, dy))
    return round(abs(angle - 180))


def main():
    WIDTH, HEIGHT = 300, 300

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Paint')
    screen.fill('white')
    
    pygame.display.flip()

    clock = pygame.time.Clock()
    FPS = 120

    LINE_LENGTH = 15
    drawing = False
    last_pos = None
    initial_pos = None
    angles = []
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                if last_pos is None:
                    initial_pos = event.pos
                    angles.append(initial_pos)
                last_pos = event.pos
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            if event.type == pygame.MOUSEMOTION and drawing:
                distance = math.sqrt((event.pos[0] - last_pos[0])**2 + (event.pos[1] - last_pos[1])**2)
                if distance > LINE_LENGTH:
                    pygame.draw.line(screen, 'black', last_pos, event.pos, 2)
                    pygame.display.flip()

                    angles.append(calculate_angle(last_pos, event.pos))
                    last_pos = event.pos

        clock.tick(FPS)

    with open(f'images/{input('Image name: ')}.img', 'w') as f:
        for angle in angles:
            f.write(str(angle) + '\n')

    pygame.quit()


if __name__ == '__main__':
    main()

