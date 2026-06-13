import numpy as np
import pygame
from UnscentedKalmanFilter import UKF
from models.crtv import transition, observation

def get_UKF_CTRV(x0, y0,sigma_a=1.0, sigma_omega=0.1, sigma_gps=0.5):

    n = 5  # state dimension
    m = 2  # observation dimension

    # --- Q: process noise ---
    # rumore entra su v e omega, propagato sulla cinematica
    # versione approssimata: diagonale sugli stati che cambiano
    Q = np.diag([
        0,              # px — nessun rumore diretto
        0,              # py — nessun rumore diretto  
        sigma_a**2,     # v  — accelerazione non modellata
        0,              # theta — cambia solo per omega
        sigma_omega**2  # omega — variazione di curvatura
    ])

    # --- R: measurement noise ---
    # GPS misura posizione con varianza sigma_gps^2
    R = np.array([
        [sigma_gps**2, 0          ],
        [0,            sigma_gps**2]
    ])

    # --- P0: incertezza iniziale ---
    P0 = np.diag([
        50,   # px — non sappiamo dove siamo
        50,   # py
        100,    # v  — velocità approssimativa nota
        10,     # theta — heading approssimativo noto
        0.1    # omega — assumiamo moto quasi rettilineo
    ])

    # --- x0: stato iniziale ---
    X0 = np.array([x0, y0, 0, 0, 0])

    kf = UKF(
        n=n, m=m,
        f=transition,
        h=observation,
        Q=Q, R=R
    )
    kf.X = X0
    kf.P = P0

    return kf


if __name__ == '__main__':
    fps = 60

    pygame.init()
    bg_color = (255, 255, 255)
    line_color_cursor = (0, 0, 0)
    line_color_kf = (255, 0, 0)

    width, height = 600, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('2D UKF CTRV Mouse Tracking')

    font = pygame.font.Font('freesansbold.ttf', 15)
    text_mouse_legend = font.render('Mouse Position', True, line_color_cursor, bg_color)
    text_kf_legend    = font.render('UKF Prediction', True, line_color_kf,     bg_color)

    text_mouse_legend_rect = text_mouse_legend.get_rect()
    text_kf_legend_rect    = text_kf_legend.get_rect()
    text_mouse_legend_rect.center = (80, 10)
    text_kf_legend_rect.center    = (80, 30)

    mouse_path = []
    kf_path    = []
    kf         = None
    dt         = 1 / fps

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        mouse_x, mouse_y = pygame.mouse.get_pos()

        if kf is None:
            kf = get_UKF_CTRV(mouse_x, mouse_y)
            continue

        kf.predict(dt)
        x_est, _ = kf.update(np.array([mouse_x, mouse_y], dtype=float))

        mouse_path.append((mouse_x, mouse_y))
        kf_path.append((float(x_est[0]), float(x_est[1])))

        screen.fill(bg_color)

        
        """
        sigma_color = (0, 0, 255)  # blu

        for i in range(kf.sigma_pred.shape[1]):
            sx = int(kf.sigma_pred[0, i])
            sy = int(kf.sigma_pred[1, i])
            pygame.draw.circle(screen, sigma_color, (sx, sy), 3)
        """
        screen.blit(text_mouse_legend, text_mouse_legend_rect)
        screen.blit(text_kf_legend,    text_kf_legend_rect)

        if len(mouse_path) > 1:
            pygame.draw.lines(screen, line_color_cursor, False, mouse_path, 8)
        if len(kf_path) > 1:
            pygame.draw.lines(screen, line_color_kf, False, kf_path, 8)

        pygame.display.flip()
        pygame.time.Clock().tick(fps)

    pygame.quit()