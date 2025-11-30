import pygame
from pygame.locals import *
from OpenGL.GL import *

from OpenGL.GLU import *
import numpy as np
import sys

# Simulation parameters
N = 128
SIZE = (N + 2) * (N + 2)
iter_solver = 4
dt = 0.1
diff = 0.0000
visc = 0.0000

# Arrays
u = np.zeros((N + 2, N + 2), dtype=np.float32)
v = np.zeros((N + 2, N + 2), dtype=np.float32)
u_prev = np.zeros((N + 2, N + 2), dtype=np.float32)
v_prev = np.zeros((N + 2, N + 2), dtype=np.float32)
dens = np.zeros((N + 2, N + 2), dtype=np.float32)
dens_prev = np.zeros((N + 2, N + 2), dtype=np.float32)

# Color arrays (R, G, B)
r = np.zeros((N + 2, N + 2), dtype=np.float32)
g = np.zeros((N + 2, N + 2), dtype=np.float32)
b = np.zeros((N + 2, N + 2), dtype=np.float32)
r_prev = np.zeros((N + 2, N + 2), dtype=np.float32)
g_prev = np.zeros((N + 2, N + 2), dtype=np.float32)
b_prev = np.zeros((N + 2, N + 2), dtype=np.float32)

def add_source(x, s, dt):
    x += dt * s

def set_bnd(b_mode, x):
    # b_mode: 0=continuous, 1=reflect x, 2=reflect y
    
    # Edges
    if b_mode == 1:
        x[0, 1:-1] = -x[1, 1:-1]
        x[-1, 1:-1] = -x[-2, 1:-1]
    else:
        x[0, 1:-1] = x[1, 1:-1]
        x[-1, 1:-1] = x[-2, 1:-1]
        
    if b_mode == 2:
        x[1:-1, 0] = -x[1:-1, 1]
        x[1:-1, -1] = -x[1:-1, -2]
    else:
        x[1:-1, 0] = x[1:-1, 1]
        x[1:-1, -1] = x[1:-1, -2]

    # Corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def lin_solve(b_mode, x, x0, a, c):
    cRecip = 1.0 / c
    for k in range(iter_solver):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[0:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, 0:-2] + x[1:-1, 2:])) * cRecip
        set_bnd(b_mode, x)

def diffuse(b_mode, x, x0, diff, dt):
    a = dt * diff * N * N
    lin_solve(b_mode, x, x0, a, 1 + 4 * a)

def advect(b_mode, d, d0, u, v, dt):
    dt0 = dt * N
    
    # Create grid of coordinates
    i, j = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing='ij')
    
    x = i - dt0 * u[1:-1, 1:-1]
    y = j - dt0 * v[1:-1, 1:-1]
    
    # Clamp
    x = np.clip(x, 0.5, N + 0.5)
    y = np.clip(y, 0.5, N + 0.5)
    
    i0 = x.astype(int)
    i1 = i0 + 1
    j0 = y.astype(int)
    j1 = j0 + 1
    
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    
    d[1:-1, 1:-1] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                     s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
                     
    set_bnd(b_mode, d)

def project(u, v, p, div):
    div[1:-1, 1:-1] = -0.5 * (u[2:, 1:-1] - u[0:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, 0:-2]) / N
    p[1:-1, 1:-1] = 0
    
    set_bnd(0, div)
    set_bnd(0, p)
    
    lin_solve(0, p, div, 1, 4)
    
    u[1:-1, 1:-1] -= 0.5 * N * (p[2:, 1:-1] - p[0:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, 0:-2])
    
    set_bnd(1, u)
    set_bnd(2, v)

def dens_step(x, x0, u, v, diff, dt):
    add_source(x, x0, dt)
    x0[:] = x[:] 
    diffuse(0, x0, x, diff, dt)
    advect(0, x, x0, u, v, dt)
    # Decay density to prevent white screen saturation
    x *= 0.995

def vorticity_confinement(u, v, dt):
    # Calculate vorticity
    du_dy = (u[:, 2:] - u[:, 0:-2]) * 0.5
    dv_dx = (v[2:, :] - v[0:-2, :]) * 0.5
    # Pad to match shape
    curl = np.zeros_like(u)
    # dv_dx is (N, N+2), covers i=1..N
    # du_dy is (N+2, N), covers j=1..N
    # We want i=1..N, j=1..N
    curl[1:-1, 1:-1] = dv_dx[:, 1:-1] - du_dy[1:-1, :]
    
    curl_abs = np.abs(curl)
    
    # Gradient of vorticity magnitude
    dw_dx = (curl_abs[2:, :] - curl_abs[0:-2, :]) * 0.5
    dw_dy = (curl_abs[:, 2:] - curl_abs[:, 0:-2]) * 0.5
    
    # Normalize
    # dw_dx is (N, N+2), dw_dy is (N+2, N)
    # We need inner (N, N)
    dw_dx_c = dw_dx[:, 1:-1]
    dw_dy_c = dw_dy[1:-1, :]
    
    length = np.sqrt(dw_dx_c**2 + dw_dy_c**2) + 1e-5
    dw_dx_c /= length
    dw_dy_c /= length
    
    # Apply force
    force = 2.0 # Confinement scale
    u[1:-1, 1:-1] += dt * force * (dw_dy_c * curl[1:-1, 1:-1])
    v[1:-1, 1:-1] += dt * force * (-dw_dx_c * curl[1:-1, 1:-1])

def vel_step(u, v, u0, v0, visc, dt):
    add_source(u, u0, dt)
    add_source(v, v0, dt)
    
    vorticity_confinement(u, v, dt)
    
    # Swap for diffusion
    u0[:], u[:] = u[:], u0[:]
    diffuse(1, u, u0, visc, dt)
    v0[:], v[:] = v[:], v0[:]
    diffuse(2, v, v0, visc, dt)
    
    project(u, v, u0, v0)
    
    # Swap for advection
    u0[:], u[:] = u[:], u0[:]
    v0[:], v[:] = v[:], v0[:]
    
    advect(1, u, u0, u0, v0, dt)
    advect(2, v, v0, u0, v0, dt)
    
    project(u, v, u0, v0)

def main():
    pygame.init()
    display = (800, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL Fluid Simulation")
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, N, 0.0, N, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Texture setup
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    clock = pygame.time.Clock()
    
    global u, v, u_prev, v_prev, r, g, b, r_prev, g_prev, b_prev
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset
                    u[:] = 0; v[:] = 0; r[:] = 0; g[:] = 0; b[:] = 0
        
        # Mouse Interaction
        mouse_pressed = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        
        # Convert mouse pos to grid coordinates
        # Pygame y is top-down, OpenGL/Grid is bottom-up usually, but let's match them
        # Here we treat grid 0,0 as top-left to match pygame for simplicity or map accordingly
        # Let's map pygame (0..800) to grid (1..N)
        
        grid_x = int((mx / display[0]) * N) + 1
        grid_y = int(((display[1] - my) / display[1]) * N) + 1 # Invert Y for OpenGL coords
        
        grid_x = max(1, min(N, grid_x))
        grid_y = max(1, min(N, grid_y))
        
        if mouse_pressed[0]: # Left click - add density and velocity
            # Add color (cycling or random)
            import time
            t = time.time()
            col_r = (np.sin(t) + 1) / 2
            col_g = (np.sin(t + 2) + 1) / 2
            col_b = (np.sin(t + 4) + 1) / 2
            
            # Add to source arrays
            r_prev[grid_x-1:grid_x+2, grid_y-1:grid_y+2] = col_r * 5
            g_prev[grid_x-1:grid_x+2, grid_y-1:grid_y+2] = col_g * 5
            b_prev[grid_x-1:grid_x+2, grid_y-1:grid_y+2] = col_b * 5
            
            # Add velocity based on mouse movement could be better, but for now just push in direction of center or random
            # Better: track mouse delta
            rel_x, rel_y = pygame.mouse.get_rel()
            # If we just call get_rel() it resets, so we might miss it if not moving. 
            # But get_rel is good for velocity.
            
            force = 5.0
            u_prev[grid_x-1:grid_x+2, grid_y-1:grid_y+2] = rel_x * force
            v_prev[grid_x-1:grid_x+2, grid_y-1:grid_y+2] = -rel_y * force # Invert Y delta
            
        else:
            # Reset sources
            r_prev[:] = 0
            g_prev[:] = 0
            b_prev[:] = 0
            u_prev[:] = 0
            v_prev[:] = 0
            pygame.mouse.get_rel() # clear rel
            
        # Step Simulation
        vel_step(u, v, u_prev, v_prev, visc, dt)
        dens_step(r, r_prev, u, v, diff, dt)
        dens_step(g, g_prev, u, v, diff, dt)
        dens_step(b, b_prev, u, v, diff, dt)
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Create RGB array for texture
        # Stack r, g, b
        # Grid is (N+2, N+2), we want inner N*N
        rgb = np.dstack((r[1:-1, 1:-1], g[1:-1, 1:-1], b[1:-1, 1:-1]))
        rgb = np.clip(rgb, 0, 1)
        
        # Transpose for OpenGL (H, W, C) -> (W, H, C) if needed, but texture expects row-major
        # Pygame/Numpy is (Row, Col) = (Y, X). OpenGL texture is (X, Y) usually? 
        # Actually glTexImage2D expects data[y][x].
        # Let's just try.
        
        # We need to convert to suitable format
        texture_data = (rgb * 255).astype(np.uint8)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, N, N, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(N, 0)
        glTexCoord2f(1, 1); glVertex2f(N, N)
        glTexCoord2f(0, 1); glVertex2f(0, N)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
