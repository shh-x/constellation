import sys
import os
import math
import time
from collections import deque
import numpy as np

# Hide Pygame support prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw
import cv2
import mediapipe as mp

def main():
    pygame.init()
    
    # ------------------
    # CONFIGURATION
    # ------------------
    WIDTH, HEIGHT = 1280, 720
    TARGET_FPS = 60
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-Body Constellation")
    clock = pygame.time.Clock()
    
    # Fonts
    pygame.font.init()
    font = pygame.font.SysFont("Courier", 32, bold=True)
    small_font = pygame.font.SysFont("Courier", 16)
    
    # Camera Background Feed
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        sys.exit()
        
    # MediaPipe Initialization
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose = PoseLandmarker.create_from_options(options)
    # ------------------
    # PHYSICS STATE
    # ------------------
    MAX_PARTICLES = 2000
    num_particles = MAX_PARTICLES
    
    def init_particles(n):
        pos = np.random.rand(n, 2) * [WIDTH, HEIGHT]
        vel = np.zeros((n, 2))
        active = np.zeros(n, dtype=bool)
        colors = np.zeros((n, 3))
        colors[:, :] = [0, 255, 255] # Default cyan
        
        proj_mode = np.zeros(n, dtype=bool)
        proj_timer = np.zeros(n)
        proj_vel = np.zeros((n, 2))
        
        return pos, vel, active, colors, proj_mode, proj_timer, proj_vel
        
    pos, vel, active, colors, proj_mode, proj_timer, proj_vel = init_particles(num_particles)
    
    # ------------------
    # RENDER SURFACES
    # ------------------
    trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    trail_surface.fill((0, 0, 0, 30)) # Alpha 30 for trail effect
    
    # ------------------
    # LOGIC STATE
    # ------------------
    person_detected = False
    scanning = False
    scan_y = 0.0
    scan_speed = HEIGHT / (TARGET_FPS * 2) # Scans across screen in 2 seconds
    
    wrist_history = deque(maxlen=10) # Track wrist positions for gesture inference
    debug_mode = False
    running = True

    print("Startup sequence complete. Entering The Void...")

    while running:
        clock.tick(TARGET_FPS)
        current_fps = clock.get_fps()
        
        # Automatic Performance Management
        if current_fps < 30 and num_particles > 500:
            num_particles -= 50
            pos = pos[:num_particles]
            vel = vel[:num_particles]
            active = active[:num_particles]
            colors = colors[:num_particles]
            proj_mode = proj_mode[:num_particles]
            proj_timer = proj_timer[:num_particles]
            proj_vel = proj_vel[:num_particles]
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Spacebar re-triggers scan manually
                    scanning = True
                    scan_y = 0.0
                    pos, vel, active, colors, proj_mode, proj_timer, proj_vel = init_particles(num_particles)
                elif event.key == pygame.K_m:
                    debug_mode = not debug_mode
                    
        # Grab and process the webcam frame
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip frame horizontally for intuitive interaction
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Feed image to MediaPipe (background only, we don't draw it unless debug mode demands)
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = pose.detect_for_video(mp_image, timestamp_ms)
        
        targets = []
        black_hole = False
        supernova = False
        target_color = [0, 255, 255]
        bh_center = None
        
        # If person detected for the first time
        if results.pose_landmarks:
            if not person_detected:
                person_detected = True
                scanning = True
                print("Biometric signature detected. Initiating scan...")
                
            landmarks = results.pose_landmarks[0]
            
            # Step 1: Filter low confidence/invisible points
            valid_dict = {}
            base_targets = []
            for i, lm in enumerate(landmarks):
                # Using 0.3 as a reasonable threshold for visibility to prevent off-camera clutter
                if getattr(lm, 'visibility', 1.0) > 0.3 and getattr(lm, 'presence', 1.0) > 0.3:
                    pt_array = np.array([lm.x * WIDTH, lm.y * HEIGHT])
                    valid_dict[i] = pt_array
                    base_targets.append(pt_array)
                    
            # Step 2: Fill out the skeleton outline via interpolation
            BODY_CONNECTIONS = [
                (11, 12),              # Shoulders
                (11, 13), (13, 15),    # Left Arm
                (12, 14), (14, 16),    # Right Arm
                (11, 23), (12, 24),    # Torso Sides
                (23, 24),              # Hips
                (23, 25), (25, 27),    # Left Leg (Thigh to Ankle)
                (24, 26), (26, 28),    # Right Leg (Thigh to Ankle)
                (27, 31), (28, 32),    # Foot tips
                # Core face outline points (Approximate)
                (3, 7), (6, 8), (9, 10), (1, 2), (0, 4) 
            ]
            
            interpolated_targets = []
            NUM_POINTS_PER_BONE = 8 # Drastically increases particle distribution along the whole body
            
            for start_idx, end_idx in BODY_CONNECTIONS:
                if start_idx in valid_dict and end_idx in valid_dict:
                    p1 = valid_dict[start_idx]
                    p2 = valid_dict[end_idx]
                    for step in range(1, NUM_POINTS_PER_BONE):
                        frac = step / NUM_POINTS_PER_BONE
                        interp_pt = p1 * (1 - frac) + p2 * frac
                        interpolated_targets.append(interp_pt)
                        
            all_targets = base_targets + interpolated_targets
            targets = np.array(all_targets) if all_targets else np.array([])
            
            # Using landmark indices from older Pose API for wrist mapping:
            # left_wrist = 15, right_wrist = 16
            LEFT_WRIST_IDX = 15
            RIGHT_WRIST_IDX = 16
            
            lw = landmarks[LEFT_WRIST_IDX]
            rw = landmarks[RIGHT_WRIST_IDX]
            
            if lw.visibility > 0.5 and rw.visibility > 0.5:
                lw_pos = np.array([lw.x * WIDTH, lw.y * HEIGHT])
                rw_pos = np.array([rw.x * WIDTH, rw.y * HEIGHT])
                wrist_history.append((lw_pos, rw_pos))
                
                wrist_dist = np.linalg.norm(lw_pos - rw_pos) / WIDTH
                
                # Black Hole Threshold (wrists close together)
                if wrist_dist < 0.1:
                    black_hole = True
                    bh_center = (lw_pos + rw_pos) / 2
                    
                # Supernova Threshold (wrists separating rapidly)
                if len(wrist_history) >= 4:
                    old_lw, old_rw = wrist_history[0]
                    new_lw, new_rw = wrist_history[-1]
                    
                    old_dist = np.linalg.norm(old_lw - old_rw)
                    new_dist = np.linalg.norm(new_lw - new_rw)
                    
                    if (new_dist - old_dist) > (WIDTH * 0.15): # 15% width expansion over history window
                        supernova = True
                        wrist_history.clear() # Prevent immediate re-trigger
                        
                # Meteor Strike Threshold (wrists moving > 25px per frame approx distance)
                if len(wrist_history) >= 4:
                    past_lw, past_rw = wrist_history[-4]
                    curr_lw, curr_rw = wrist_history[-1]
                    
                    lw_vel_vec = curr_lw - past_lw
                    rw_vel_vec = curr_rw - past_rw
                    
                    lw_speed = np.linalg.norm(lw_vel_vec)
                    rw_speed = np.linalg.norm(rw_vel_vec)
                    
                    STRIKE_THRESHOLD = 75.0 # (25px * 3 frames)
                    STAR_COUNT = 20
                    
                    strikes = []
                    if lw_speed > STRIKE_THRESHOLD:
                        strikes.append((curr_lw, lw_vel_vec / (lw_speed + 1e-5)))
                    if rw_speed > STRIKE_THRESHOLD:
                        strikes.append((curr_rw, rw_vel_vec / (rw_speed + 1e-5)))
                        
                    for strike_pos, strike_dir in strikes:
                        eligible = active & (~proj_mode)
                        if np.any(eligible):
                            eligible_indices = np.where(eligible)[0]
                            eligible_pos = pos[eligible_indices]
                            dists_sq = np.sum((eligible_pos - strike_pos)**2, axis=1)
                            
                            sort_idx = np.argsort(dists_sq)
                            closest_idx = eligible_indices[sort_idx[:STAR_COUNT]]
                            
                            proj_mode[closest_idx] = True
                            proj_timer[closest_idx] = 2.0 * TARGET_FPS
                            proj_vel[closest_idx] = strike_dir * 40.0 # Fast constant velocity
                            colors[closest_idx] = [255, 165, 0] # Orange / Gold
                            vel[closest_idx] = [0, 0] # Reset normal velocity
                        
                # Zenith/Nadir Color Shift Mapping
                avg_y = (lw_pos[1] + rw_pos[1]) / 2
                norm_y = np.clip(avg_y / HEIGHT, 0, 1)
                
                # Lerp from Golden/White (high) to Deep Purple/Red (low)
                r = int(255 * (1 - norm_y) + 128 * norm_y)
                g = int(230 * (1 - norm_y) +   0 * norm_y)
                b = int(100 * (1 - norm_y) + 128 * norm_y)
                target_color = [r, g, b]
                
        # Semi-transparent trail wipe over background
        screen.blit(trail_surface, (0, 0))
        
        # ------------------
        # PHYSICS EVALUATION
        # ------------------
        G_BASE = 5000.0
        DAMPING = 0.95
        
        if supernova:
            screen.fill((255, 255, 255)) # Visual Flash
            G = -30000.0 # Violent negative repulsion
            target_color = [255, 100, 100]
        elif black_hole:
            G = G_BASE * 10.0 # Multiply mass drastically
            targets = np.array([bh_center])
            target_color = [255, 255, 255] # White hot core
            
            # Subtle visual draw-in effect for the background
            pygame.draw.circle(screen, (10, 10, 10), (int(bh_center[0]), int(bh_center[1])), 500, 20)
        else:
            G = G_BASE

        # Scan Bar Sequence
        if scanning:
            scan_y += scan_speed
            if scan_y >= HEIGHT:
                scanning = False
                
            # Bring stars to life dynamically as bar drops
            mask = pos[:, 1] < scan_y
            active[mask] = True
            
            # Render visual bar
            pygame.draw.line(screen, (0, 255, 255), (0, int(scan_y)), (WIDTH, int(scan_y)), 3)
            
            # Display Holographic Text Effect
            if time.time() % 0.4 < 0.2: # Simple flicker
                if len(targets) > 0 and results.pose_landmarks:
                    NOSE_IDX = 0
                    nose = landmarks[NOSE_IDX]
                    nx, ny = int(nose.x * WIDTH), int(nose.y * HEIGHT)
                    t1 = font.render("SCANNING BIOMETRIC SIGNATURE...", True, (0, 255, 0))
                    t2 = small_font.render("STELLAR MASS CALCULATED.", True, (0, 255, 0))
                    screen.blit(t1, (nx + 20, ny - 30))
                    screen.blit(t2, (nx + 20, ny + 10))
                else:
                    t = font.render("SEARCHING VOID...", True, (0, 255, 0))
                    screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2))

        # N-Body Simulation Math
        normal_mask = active & (~proj_mode)
        
        # 1. Projectile Logic
        proj_mask = proj_mode & active
        if np.any(proj_mask):
            pos[proj_mask] += proj_vel[proj_mask]
            proj_timer[proj_mask] -= 1
            
            # Reset if off-screen or timer passes 2 seconds
            hit_edge = (pos[:, 0] < 0) | (pos[:, 0] >= WIDTH) | (pos[:, 1] < 0) | (pos[:, 1] >= HEIGHT)
            reset_cond = proj_mask & ((proj_timer <= 0) | hit_edge)
            if np.any(reset_cond):
                proj_mode[reset_cond] = False
                
        # Re-evaluate normal mask gracefully re-integrating reset projectiles
        normal_mask = active & (~proj_mode)
        
        if len(targets) > 0 and (not scanning or np.any(normal_mask)):
            active_pos = pos[normal_mask]
            active_vel = vel[normal_mask]
            
            if len(active_pos) > 0:
                if len(targets) == 1: 
                    # Singular core focus (Black hole behavior)
                    diffs = targets[0] - active_pos
                    dists_sq = np.sum(diffs**2, axis=1) + 1.0
                    dists = np.sqrt(dists_sq)
                    force_mag = G / dists_sq
                    acc = (diffs / dists[:, np.newaxis]) * force_mag[:, np.newaxis]
                else:
                    # Broadcasting 2D pos to multiple targets per particle
                    diffs = targets[np.newaxis, :, :] - active_pos[:, np.newaxis, :]
                    dists_sq = np.sum(diffs**2, axis=2)
                    
                    closest_idx = np.argmin(dists_sq, axis=1)
                    
                    closest_diff = diffs[np.arange(len(active_pos)), closest_idx]
                    # Epsilon 20.0 prevents singularities when orbits pass through targets
                    closest_dist_sq = dists_sq[np.arange(len(active_pos)), closest_idx] + 20.0 
                    closest_dist = np.sqrt(closest_dist_sq)
                    
                    if supernova:
                        # Keep standard inverse-square repulsion for the explosion
                        force_mag = G / closest_dist_sq
                    else:
                        AURA_RADIUS = 40.0
                        error = closest_dist - AURA_RADIUS
                        # Proportional spring force to pull into the aura outline, clipped for stability
                        force_mag = np.clip(0.15 * error, -10.0, 10.0)
                        # Add light brownian motion / shimmer
                        force_mag += np.random.randn(len(active_pos)) * 0.5
                    
                    acc = (closest_diff / closest_dist[:, np.newaxis]) * force_mag[:, np.newaxis]
                    
                active_vel += acc
                active_vel *= DAMPING
                active_pos += active_vel
                
                pos[normal_mask] = active_pos
                vel[normal_mask] = active_vel
                
        # Smoothly lerp towards gestural target colors (Naturally creates a 0.5s fade back to body color)
        colors[normal_mask] = colors[normal_mask] * 0.90 + np.array(target_color) * 0.10
                
        # Render Point Data
        active_indices = np.where(active)[0]
        for i in active_indices:
            x, y = int(pos[i, 0]), int(pos[i, 1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                c = (min(255, max(0, int(colors[i, 0]))),
                     min(255, max(0, int(colors[i, 1]))),
                     min(255, max(0, int(colors[i, 2]))))
                try:
                    pygame.gfxdraw.aacircle(screen, x, y, 2, c)
                    pygame.gfxdraw.filled_circle(screen, x, y, 2, c)
                except ValueError:
                    pass
                
        # Debug Skeleton Layout
        if debug_mode and len(targets) > 0:
            for tx, ty in targets:
                pygame.draw.circle(screen, (255, 0, 0), (int(tx), int(ty)), 4)
                
        # OSD Display
        fps_surf = small_font.render(f"FPS: {int(current_fps)} | Particles: {num_particles}", True, (255, 255, 255))
        screen.blit(fps_surf, (10, 10))
        controls_surf = small_font.render("SPACE: Scan | M: Debug | ESC: Quit", True, (150, 150, 150))
        screen.blit(controls_surf, (10, HEIGHT - 30))

        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
