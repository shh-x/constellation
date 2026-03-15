"""
╔══════════════════════════════════════════════════════╗
║        COSMIC SIMULATION ENGINE  — v5.0              ║
║                                                      ║
║  GESTURE ZONES (based on wrist distance only):       ║
║  ─────────────────────────────────────────────────   ║
║  dist < 0.16  → BLACK HOLE   (hands together)        ║
║  dist > 0.55  → SUPERNOVA    (hands apart, once)     ║
║  hands above nose → GALAXY SWIRL                     ║
║  otherwise    → STELLAR DRIFT                        ║
║                                                      ║
║  BLACK HOLE:  particles drain into hands forever     ║
║               stays until SPACE is pressed           ║
║  SUPERNOVA:   fires once, 2.5s cooldown              ║
╚══════════════════════════════════════════════════════╝
Controls:
  SPACE  — rescan / respawn particles on skeleton
  M      — debug skeleton overlay
  S      — screenshot
  ESC    — quit
"""

import sys, os, math, time
# Windows: prevent Unicode crash in terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from collections import deque
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw
import cv2
import mediapipe as mp

# ===================================================
#  CONSTANTS
# ===================================================
WIDTH, HEIGHT  = 1280, 720
TARGET_FPS     = 60
MAX_PARTICLES  = 2000

_HERE         = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(_HERE, "background.png")

MODE_NORMAL    = "STELLAR DRIFT"
MODE_BLACKHOLE = "BLACK HOLE"
MODE_SUPERNOVA = "SUPERNOVA"
MODE_GALAXY    = "GALAXY SWIRL"

# Gesture distance thresholds (fraction of shoulder span)
DIST_BH_THRESH = 0.38   # closer than this  → black hole
DIST_SN_THRESH = 0.55   # farther than this → supernova trigger
BH_HOLD_FRAMES = 1      # 1 = triggers on the VERY FIRST frame hands are close

def clamp(v, lo, hi):  return max(lo, min(hi, v))
def lerp(a, b, t):     return a + (b - a) * t
def lerp_col(a, b, t): return tuple(int(lerp(a[i], b[i], t)) for i in range(3))


# ===================================================
#  SOUND MANAGER
# ===================================================
class SoundManager:
    SR = 44100

    def __init__(self):
        self._ok = False
        try:
            pygame.mixer.pre_init(self.SR, -16, 2, 256)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self._sounds = {}
            self._build_all()
            self._ok = True
            print("  [Audio] OK")
        except Exception as e:
            print(f"  [Audio] Failed: {e}")

    def _t(self, dur):
        return np.linspace(0, dur, int(self.SR * dur), endpoint=False)

    def _env(self, arr, attack=0.02, release=0.08):
        n = len(arr)
        a = min(int(self.SR * attack), n // 4)
        r = min(int(self.SR * release), n // 4)
        env = np.ones(n)
        if a: env[:a]  = np.linspace(0, 1, a)
        if r: env[-r:] = np.linspace(1, 0, r)
        return arr * env

    def _make(self, arr):
        arr = np.clip(arr, -1, 1)
        i16 = (arr * 28000).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack([i16, i16]))

    def _build_all(self):
        # Ambient space hum
        t = self._t(2.0)
        d = np.sin(2*np.pi*55*t)*0.5 + np.sin(2*np.pi*110*t)*0.25 + np.sin(2*np.pi*82.5*t)*0.2
        self._sounds['ambient'] = self._make(self._env(d, 0.4, 0.4) * 0.3)

        # Black hole: deep falling rumble
        t = self._t(1.4)
        freq = 80 * np.exp(-t * 0.8)
        bh   = np.sin(2*np.pi * np.cumsum(freq) / self.SR)
        bh  += np.sin(2*np.pi*40*t)*0.4 + np.random.randn(len(t))*0.04
        self._sounds['blackhole'] = self._make(self._env(bh, 0.05, 0.4) * 0.65)

        # Supernova: crack + boom
        t  = self._t(0.6)
        sn = (np.random.randn(len(t))*np.exp(-t*12)*0.6 +
              np.sin(2*np.pi*400*t)*np.exp(-t*8)*0.3 +
              (np.sin(2*np.pi*60*t)+np.sin(2*np.pi*90*t))*np.exp(-t*5)*0.5)
        self._sounds['supernova'] = self._make(self._env(sn, 0.001, 0.15) * 0.85)

        # Galaxy: ethereal shimmer
        t     = self._t(1.2)
        freq2 = 200 + t*300
        gal   = np.sin(2*np.pi*np.cumsum(freq2)/self.SR)
        gal  += np.sin(2*np.pi*660*t)*0.3*np.sin(2*np.pi*3*t)
        self._sounds['galaxy'] = self._make(self._env(gal, 0.05, 0.5) * 0.5)

        # Scan ping
        t    = self._t(0.35)
        scan = np.sin(2*np.pi*1200*t)*np.exp(-t*6) + np.sin(2*np.pi*1800*t)*np.exp(-t*10)*0.4
        self._sounds['scan'] = self._make(self._env(scan, 0.005, 0.1) * 0.7)

        self._amb_ch = pygame.mixer.Channel(0)
        self._amb_ch.set_volume(0.18)
        self._amb_ch.play(self._sounds['ambient'], loops=-1)

    def play(self, name, vol=0.7):
        if not self._ok: return
        s = self._sounds.get(name)
        if not s: return
        ch = pygame.mixer.find_channel(True)
        if ch:
            ch.set_volume(vol)
            ch.play(s)


# ===================================================
#  GESTURE RECOGNISER  v5
#
#  Pure distance-zone logic — no velocity, no latch:
#
#  Zone A  dist_norm < DIST_BH_THRESH  →  Black Hole
#  Zone B  dist_norm > DIST_SN_THRESH  →  Supernova (once per cooldown)
#  Zone C  hands above nose            →  Galaxy Swirl
#  Zone D  everything else             →  Normal
#
#  Black hole activates after BH_HOLD_FRAMES consecutive
#  frames in Zone A, deactivates immediately on exit.
#
#  Supernova fires ONCE when entering Zone B.
#  After firing, a 2.5s cooldown prevents re-fire
#  even if hands stay apart.
# ===================================================
class GestureRecogniser:
    SUPERNOVA_COOLDOWN = 2.5

    def __init__(self, w, h):
        self.w, self.h     = w, h
        self._slw          = None      # smoothed left  wrist
        self._srw          = None      # smoothed right wrist
        self._K            = 0.70      # smoothing factor (high = fast response)
        self._bh_frames    = 0         # consecutive Zone-A frames
        self._sn_cooldown  = 0.0       # supernova cooldown timer
        self._was_in_zone_b = False    # True while hands are apart

    def _get(self, lm, idx, min_vis=0.15):
        l   = lm[idx]
        vis = getattr(l, 'visibility', 1.0)
        pre = getattr(l, 'presence',   1.0)
        if vis < min_vis or pre < min_vis:
            return None
        return np.array([l.x * self.w, l.y * self.h])

    def update(self, landmarks, dt):
        """
        Returns (mode, bh_center_or_None, supernova_fired_bool)
        """
        self._sn_cooldown = max(0.0, self._sn_cooldown - dt)

        if landmarks is None:
            self._slw = self._srw = None
            self._bh_frames = 0
            self._was_in_zone_b = False
            return MODE_NORMAL, None, False

        lm  = landmarks
        lw  = self._get(lm, 15)          # left  wrist
        rw  = self._get(lm, 16)          # right wrist
        nse = self._get(lm,  0, min_vis=0.15)
        ls  = self._get(lm, 11)          # left  shoulder
        rs  = self._get(lm, 12)          # right shoulder

        # When hands overlap MediaPipe often loses wrists — use last smooth pos
        if lw is None:
            lw = self._slw  # fallback to last known
        if rw is None:
            rw = self._srw
        if lw is None or rw is None:
            self._bh_frames = 0
            return MODE_NORMAL, None, False

        # -- Smooth wrist positions
        K = self._K
        if self._slw is None:
            self._slw, self._srw = lw.copy(), rw.copy()
        else:
            self._slw = self._slw*(1-K) + lw*K
            self._srw = self._srw*(1-K) + rw*K
        slw, srw = self._slw, self._srw

        # -- Body-scale reference
        if ls is not None and rs is not None:
            body_ref = max(np.linalg.norm(ls - rs), 1.0)
        else:
            body_ref = self.w * 0.25

        dist_smooth = np.linalg.norm(slw - srw) / body_ref
        dist_raw    = np.linalg.norm(lw - rw) / body_ref
        # Use whichever is smaller — so the INSTANT raw hands are close, it triggers
        dist_norm   = min(dist_smooth, dist_raw)

        # Debug: print distance so user can see how close they are
        self._dbg_counter = getattr(self, '_dbg_counter', 0) + 1
        if self._dbg_counter % 15 == 0:
            zone = "BH" if dist_norm < DIST_BH_THRESH else ("SN" if dist_norm > DIST_SN_THRESH else "--")
            print(f"  [GESTURE] raw={dist_raw:.2f} smooth={dist_smooth:.2f} -> {dist_norm:.2f}  BH<{DIST_BH_THRESH}  zone={zone}")

        # ====================================
        #  ZONE A — HANDS TOGETHER → Black Hole
        # ====================================
        if dist_norm < DIST_BH_THRESH:
            self._bh_frames    += 1
            self._was_in_zone_b = False
            if self._bh_frames >= BH_HOLD_FRAMES:
                # Use midpoint of RAW wrists for accuracy
                bh_pos = (lw + rw) / 2
                return MODE_BLACKHOLE, bh_pos, False
            return MODE_NORMAL, None, False

        # Hands not together — decay bh counter
        self._bh_frames = max(0, self._bh_frames - 1)

        # ====================================
        #  ZONE B — HANDS APART → Supernova
        # ====================================
        if dist_norm > DIST_SN_THRESH:
            supernova_fired = False
            # Fire once on ENTRY into Zone B (was_in_zone_b was False)
            if not self._was_in_zone_b and self._sn_cooldown <= 0:
                supernova_fired     = True
                self._sn_cooldown   = self.SUPERNOVA_COOLDOWN
            self._was_in_zone_b = True
            return MODE_SUPERNOVA if supernova_fired else MODE_NORMAL, None, supernova_fired

        # Between zones — reset zone-B latch so next separation fires again
        self._was_in_zone_b = False

        # ====================================
        #  ZONE C — HANDS UP → Galaxy Swirl
        # ====================================
        if nse is not None:
            above = slw[1] < nse[1] - 30 and srw[1] < nse[1] - 30
        elif ls is not None and rs is not None:
            above = slw[1] < (ls[1]+rs[1])/2 and srw[1] < (ls[1]+rs[1])/2
        else:
            above = False

        if above:
            return MODE_GALAXY, None, False

        # ====================================
        #  ZONE D — Normal
        # ====================================
        return MODE_NORMAL, None, False


# ===================================================
#  PARTICLE SPAWNER — on skeleton
# ===================================================
def spawn_on_skeleton(num_p, targets):
    if targets is None or len(targets) == 0:
        return np.random.rand(num_p, 2) * [WIDTH, HEIGHT]
    idx    = np.random.randint(0, len(targets), num_p)
    pos    = targets[idx].copy() + np.random.randn(num_p, 2) * 20.0
    pos[:,0] = np.clip(pos[:,0], 0, WIDTH-1)
    pos[:,1] = np.clip(pos[:,1], 0, HEIGHT-1)
    return pos


# ===================================================
#  ACCRETION DISK
# ===================================================
class AccretionDisk:
    def __init__(self):
        self.alpha  = 0.0
        self.center = np.array([WIDTH/2.0, HEIGHT/2.0])

    def update(self, active, center, dt):
        tgt        = 1.0 if active else 0.0
        self.alpha = clamp(self.alpha + (tgt - self.alpha)*dt*35, 0, 1)
        if active:
            self.center = center.copy()

    def draw(self, surf):
        if self.alpha <= 0.01: return
        cx, cy = int(self.center[0]), int(self.center[1])
        s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
        a = self.alpha
        # Glowing rings
        for r, th, col in [
            (140,24,(255, 90, 10)),
            (108,16,(255,170, 30)),
            ( 76,11,(255,240,110)),
            ( 48, 7,(255,255,210)),
        ]:
            pygame.draw.ellipse(s, (*col, int(clamp(190*a,0,255))),
                                (cx-r, cy-r//3, r*2, r*2//3), th)
        pygame.draw.circle(s, (0,0,0,       int(255*a)), (cx,cy), 32)
        pygame.draw.circle(s, (200,200,255, int(140*a)), (cx,cy), 40, 2)
        for gr, ga in [(22,80),(16,120),(10,180)]:
            pygame.draw.circle(s, (255,140,0,int(ga*a)), (cx,cy), gr)
        surf.blit(s, (0,0))


# ===================================================
#  SHOCKWAVE
# ===================================================
class ShockwaveManager:
    def __init__(self): self.waves = []

    def trigger(self, cx, cy):
        self.waves += [
            {'cx':cx,'cy':cy,'r':4,  'maxr':1100,'alpha':255,'col':(255,210, 80),'spd':950},
            {'cx':cx,'cy':cy,'r':2,  'maxr': 850,'alpha':200,'col':(120,180,255),'spd':750},
            {'cx':cx,'cy':cy,'r':1,  'maxr': 600,'alpha':160,'col':(255,255,255),'spd':600},
        ]

    def update(self, dt):
        for w in self.waves:
            w['r']    += dt * w['spd']
            w['alpha'] = max(0, w['alpha'] - dt*300)
        self.waves = [w for w in self.waves if w['alpha'] > 0]

    def draw(self, surf):
        if not self.waves: return
        s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
        for w in self.waves:
            a  = int(clamp(w['alpha'],0,255))
            r  = int(w['r'])
            if r <= 0: continue
            th = max(1, int(5*(1-w['r']/w['maxr']))+1)
            pygame.draw.circle(s, (*w['col'],a), (int(w['cx']),int(w['cy'])), r, th)
        surf.blit(s, (0,0))


# ===================================================
#  GALAXY OVERLAY
# ===================================================
class GalaxyOverlay:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.alpha     = 0.0
        self.angle     = 0.0
        self._arms     = self._gen()

    def _gen(self):
        arms = []
        for arm in range(3):
            base = arm*(2*math.pi/3)
            pts  = []
            for i in range(60):
                r   = 30 + i*4.5
                ang = base + i*0.18
                pts.append((WIDTH//2+r*math.cos(ang), HEIGHT//2+r*math.sin(ang)*0.5, r))
            arms.append(pts)
        return arms

    def update(self, active, dt):
        self.angle += dt*40
        tgt = 1.0 if active else 0.0
        self.alpha = clamp(self.alpha+(tgt-self.alpha)*dt*30, 0, 1)

    def draw(self, surf):
        if self.alpha <= 0.01: return
        s  = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        t  = time.time()
        ar = math.radians(self.angle)
        cx, cy = self.w//2, self.h//2
        for arm_pts in self._arms:
            for i,(x,y,r) in enumerate(arm_pts):
                frac = i/len(arm_pts)
                dx,dy = x-cx, y-cy
                nx = cx+dx*math.cos(ar)-dy*math.sin(ar)
                ny = cy+dx*math.sin(ar)+dy*math.cos(ar)
                a   = int(clamp(130*self.alpha*(1-frac*0.6),0,255))
                pulse = abs(math.sin(t*3+frac*8))
                col = lerp_col((80,40,200),(180,120,255),pulse)
                pygame.draw.circle(s,(*col,a),(int(nx),int(ny)),max(1,int(2*(1-frac*0.5))))
        surf.blit(s,(0,0))


# ===================================================
#  HUD
# ===================================================
class HUD:
    COL_CYAN   = (100, 220, 255)
    COL_PURPLE = (160,  90, 255)
    COL_WHITE  = (230, 235, 255)
    COL_GOLD   = (255, 200,  60)
    COL_RED    = (255,  80,  70)
    COL_PANEL  = (10,   18,  40, 165)

    def __init__(self, w, h):
        self.w, self.h      = w, h
        self._t             = 0.0
        self.flash_msg      = ""
        self.flash_timer    = 0.0
        self.flash_col      = (100, 220, 255)
        self.screenshot_msg = 0.0
        self.mode           = MODE_NORMAL
        self.fps            = 60
        self.particles      = MAX_PARTICLES
        self.person         = False
        self.scan_pct       = 0.0
        self.scanning       = False

        pygame.font.init()
        self._f_flash = pygame.font.SysFont("Courier New", 52, bold=True)
        self._f_med   = pygame.font.SysFont("Courier New", 20, bold=True)
        self._f_small = pygame.font.SysFont("Courier New", 13)
        self._f_tiny  = pygame.font.SysFont("Courier New", 11)

    def update(self, dt, mode, fps, particles, person, scanning, scan_y):
        self._t             += dt
        self.mode            = mode
        self.fps             = fps
        self.particles       = particles
        self.person          = person
        self.scanning        = scanning
        self.scan_pct        = scan_y / HEIGHT
        if self.flash_timer    > 0: self.flash_timer    -= dt
        if self.screenshot_msg > 0: self.screenshot_msg -= dt

    def flash(self, msg, col, dur=2.2):
        self.flash_msg   = msg
        self.flash_col   = col
        self.flash_timer = dur

    def _mc(self):
        return {
            MODE_BLACKHOLE: self.COL_GOLD,
            MODE_SUPERNOVA: self.COL_RED,
            MODE_GALAXY:    self.COL_PURPLE,
        }.get(self.mode, self.COL_CYAN)

    def _panel(self, surf, x, y, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill(self.COL_PANEL)
        surf.blit(s, (x, y))
        mc = self._mc()
        pygame.draw.rect(surf, (*mc, 55), (x, y, w, h), 1)

    def draw(self, surf):
        ov    = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        mc    = self._mc()
        pulse = 0.72 + 0.28*abs(math.sin(self._t*2.6))

        # Corner brackets
        CL = 32
        for (cx2,cy2),(dx,dy) in [
            ((8,8),(1,1)),((self.w-8,8),(-1,1)),
            ((8,self.h-8),(1,-1)),((self.w-8,self.h-8),(-1,-1))
        ]:
            ca = int(210*pulse)
            pygame.draw.line(ov,(*mc,ca),(cx2,cy2),(cx2+dx*CL,cy2),2)
            pygame.draw.line(ov,(*mc,ca),(cx2,cy2),(cx2,cy2+dy*CL),2)
            pygame.draw.circle(ov,(*mc,ca),(cx2,cy2),2)

        # Top-left status panel
        self._panel(ov, 8, 8, 280, 58)
        dot_col = (60,220,80) if self.person else (180,60,60)
        pygame.draw.circle(ov, (*dot_col,220), (22,22), 5)
        ring_r = int(8+3*abs(math.sin(self._t*4)))
        pygame.draw.circle(ov, (*dot_col,int(80*pulse)), (22,22), ring_r, 1)
        st = self._f_small.render(
            "ASTRAL ENTITY LOCKED" if self.person else "SCANNING VOID...",
            True, (*dot_col,210))
        ov.blit(st, (32,15))
        ov.blit(self._f_tiny.render(
            f"FPS {int(self.fps):>3}   STARS {self.particles:>4}",
            True, (*self.COL_WHITE,130)), (14,38))

        # Top-right mode badge
        badge = self._f_med.render(self.mode, True, (*mc,int(235*pulse)))
        bw    = badge.get_width()
        bx    = self.w - bw - 16
        self._panel(ov, bx-8, 6, bw+16, 44)
        ov.blit(badge, (bx,12))
        bar_fill = int(bw*abs(math.sin(self._t*1.6)))
        pygame.draw.rect(ov, (*mc,50),  (bx,38,bw,2))
        pygame.draw.rect(ov, (*mc,220), (bx,38,bar_fill,2))

        # Gesture guide (bottom-left)
        gestures = [
            ("Hands Together", "Black Hole"),
            ("Hands Up",       "Galaxy Swirl"),
            ("Hands Apart",    "Supernova"),
        ]
        gh = 14*len(gestures)+16
        gy = self.h - gh - 28
        self._panel(ov, 8, gy, 240, gh)
        for i,(trigger,effect) in enumerate(gestures):
            active_g = effect.upper().replace(" ","") in self.mode.upper().replace(" ","")
            col_e    = (*mc,210) if active_g else (120,140,170,170)
            ov.blit(self._f_tiny.render(f"{trigger}  ->  {effect}", True, col_e),
                    (16, gy+8+i*14))

        # Bottom controls
        ov.blit(self._f_tiny.render(
            "SPACE Rescan  |  M Debug  |  S Screenshot  |  ESC Quit",
            True,(70,85,110,135)),
            (self.w//2 - self._f_tiny.size(
                "SPACE Rescan  |  M Debug  |  S Screenshot  |  ESC Quit")[0]//2,
             self.h-16))

        # Scan bar
        if self.scanning:
            bw2   = self.w - 40
            fill2 = int(bw2*self.scan_pct)
            pygame.draw.rect(ov, (*mc,35),  (20,self.h-46,bw2,3))
            pygame.draw.rect(ov, (*mc,200), (20,self.h-46,fill2,3))
            if fill2 > 0:
                pygame.draw.circle(ov,(*mc,180),(20+fill2,self.h-44),4)
            ov.blit(self._f_small.render(
                "ASTRAL ENTITY SCAN IN PROGRESS", True,(60,220,80,180)),
                (self.w//2-self._f_small.size(
                    "ASTRAL ENTITY SCAN IN PROGRESS")[0]//2, self.h-62))

        # Centre flash pop
        if self.flash_timer > 0:
            alpha = int(255 * min(1.0, self.flash_timer / 0.5))
            fc    = self.flash_col

            for off in [(3,3),(-3,3),(3,-3),(-3,-3)]:
                g = self._f_flash.render(self.flash_msg, True,
                                         (*fc, clamp(alpha//5, 0, 255)))
                ov.blit(g, (self.w//2 - g.get_width()//2 + off[0],
                            self.h//2 - g.get_height()//2 - 60 + off[1]))

            fs = self._f_flash.render(self.flash_msg, True,
                                      (*fc, clamp(alpha, 0, 255)))
            ov.blit(fs, (self.w//2 - fs.get_width()//2,
                         self.h//2 - fs.get_height()//2 - 60))

            tw = fs.get_width()
            tx = self.w//2 - tw//2
            ty = self.h//2 - 60 + fs.get_height() + 4
            line_fill = int(tw * min(1.0, (2.2 - self.flash_timer) / 0.4))
            pygame.draw.rect(ov, (*fc, clamp(alpha//2, 0, 255)),
                             (tx, ty, tw, 2))
            pygame.draw.rect(ov, (*fc, clamp(alpha, 0, 255)),
                             (tx, ty, line_fill, 2))

        # Screenshot toast
        if self.screenshot_msg > 0:
            a = int(clamp(self.screenshot_msg*130,0,255))
            self._panel(ov,self.w//2-130,60,260,36)
            ss = self._f_med.render("* SCREENSHOT SAVED *",True,(*self.COL_GOLD,a))
            ov.blit(ss,(self.w//2-ss.get_width()//2,68))

        surf.blit(ov,(0,0))


# ===================================================
#  MAIN
# ===================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("COSMIC SIMULATION ENGINE")
    clock  = pygame.time.Clock()

    print("=" * 54)
    print("  COSMIC SIMULATION ENGINE  v5.0")
    print("=" * 54)

    # Background
    print("  [1/6] Loading background ...")
    try:
        bg = pygame.transform.scale(
            pygame.image.load(BG_IMAGE_PATH).convert(), (WIDTH, HEIGHT))
        print(f"        OK")
    except Exception as e:
        print(f"        WARNING: {e} - solid fallback")
        bg = pygame.Surface((WIDTH, HEIGHT)); bg.fill((2, 6, 22))

    # Webcam
    print("  [2/6] Opening webcam ...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("  ERROR: webcam unavailable."); sys.exit(1)

    # MediaPipe
    print("  [3/6] Loading pose model ...")
    model_path            = os.path.join(_HERE, "pose_landmarker.task")
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    pose = PoseLandmarker.create_from_options(options)

    # Subsystems
    print("  [4/6] Creating subsystems ...")
    gesture    = GestureRecogniser(WIDTH, HEIGHT)
    accretion  = AccretionDisk()
    shockwaves = ShockwaveManager()
    galaxy_ov  = GalaxyOverlay(WIDTH, HEIGHT)
    hud        = HUD(WIDTH, HEIGHT)

    print("  [5/6] Building audio ...")
    sound = SoundManager()

    # Particles
    print("  [6/6] Initialising particles ...")
    num_p  = MAX_PARTICLES
    pos    = np.random.rand(num_p, 2) * [WIDTH, HEIGHT]
    vel    = np.zeros((num_p, 2))
    active = np.zeros(num_p, dtype=bool)
    colors = np.full((num_p, 3), [0.0, 200.0, 255.0])

    BODY_CONNECTIONS = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(25,27),
        (24,26),(26,28),(27,31),(28,32),
        (3,7),(6,8),(9,10),(1,2),(0,4),
    ]
    NUM_INTERP = 8

    # Simulation state
    mode             = MODE_NORMAL
    last_mode        = mode
    person           = False
    scanning         = False
    scan_y           = 0.0
    scan_speed       = HEIGHT / (TARGET_FPS * 2.2)
    targets          = np.array([])
    tgt_col          = [0.0, 200.0, 255.0]
    bh_center        = None
    debug_mode       = False
    flash_white      = 0.0
    prev_t           = time.time()
    running          = True
    _respawn_pending = False

    # Persistent trail surface
    trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # ── BLACK HOLE CONSTANTS ──────────────────────────────────────────────
    # G value high enough that ALL particles spiral in within a few seconds.
    # Drain radius: particles closer than this are consumed each frame.
    BH_G           = 180000.0   # was 65000 — strong enough to overcome any orbit
    BH_DAMPING     = 0.88       # was 0.92 — less damping = faster acceleration
    BH_CORE_RADIUS = 28         # was 18   — wider drain mouth
    BH_SWIRL       = 0.18       # tangential spin fraction (spiral effect)
    BH_SPEED_CAP   = 900.0      # px/s cap to prevent teleportation

    print("=" * 54)
    print("  READY - Step into the void.")
    print("  SPACE to respawn particles after black hole.")
    print("=" * 54)

    while running:
        now    = time.time()
        dt     = clamp(now - prev_t, 0.001, 0.05)
        prev_t = now
        clock.tick(TARGET_FPS)
        fps = clock.get_fps()

        # Auto-reduce particles if FPS drops
        if fps < 26 and num_p > 400:
            cut    = min(40, num_p-400)
            num_p -= cut
            pos    = pos[:num_p];    vel    = vel[:num_p]
            active = active[:num_p]; colors = colors[:num_p]

        # ── Events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    scanning         = True
                    scan_y           = 0.0
                    active[:]        = False
                    vel[:]           = 0
                    _respawn_pending = True
                    sound.play('scan')

                elif event.key == pygame.K_m:
                    debug_mode = not debug_mode

                elif event.key == pygame.K_s:
                    fname = f"cosmic_{int(time.time())}.png"
                    pygame.image.save(screen, fname)
                    hud.screenshot_msg = 2.8
                    print(f"  Screenshot -> {fname}")

        # ── Webcam + MediaPipe ──────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms     = int(time.time() * 1000)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results   = pose.detect_for_video(mp_img, ts_ms)

        bh_center     = None
        supernova_now = False
        targets       = np.array([])

        if results.pose_landmarks:
            # First-ever detection → trigger scan
            if not person:
                person           = True
                scanning         = True
                scan_y           = 0.0
                active[:]        = False
                _respawn_pending = True
                sound.play('scan')
                hud.flash("ASTRAL ENTITY DETECTED", (60, 220, 80), 2.5)
                print("  Astral entity detected.")

            lms = results.pose_landmarks[0]
            mode, bh_center, supernova_now = gesture.update(lms, dt)

            # Particle colour from average wrist height
            try:
                lw16 = lms[15]; rw16 = lms[16]
                if getattr(lw16,'visibility',0)>0.4 and getattr(rw16,'visibility',0)>0.4:
                    n_y     = clamp((lw16.y+rw16.y)/2, 0, 1)
                    tgt_col = [int(lerp(0,200,n_y)), int(lerp(200,80,n_y)), int(lerp(255,255,n_y))]
            except Exception:
                pass

            # Build skeleton target cloud
            valid = {}
            base  = []
            for i, lm in enumerate(lms):
                if getattr(lm,'visibility',1.0)>0.3 and getattr(lm,'presence',1.0)>0.3:
                    pt = np.array([lm.x*WIDTH, lm.y*HEIGHT])
                    valid[i] = pt
                    base.append(pt)
            interp = []
            for si, ei in BODY_CONNECTIONS:
                if si in valid and ei in valid:
                    p1, p2 = valid[si], valid[ei]
                    for step in range(1, NUM_INTERP):
                        f = step/NUM_INTERP
                        interp.append(p1*(1-f)+p2*f)
            all_pts = base + interp
            targets = np.array(all_pts) if all_pts else np.array([])

        else:
            mode = MODE_NORMAL
            gesture.update(None, dt)

        # ── Mode-change events ──────────────────────────────────────────
        if mode != last_mode:
            prev_mode = last_mode
            last_mode = mode

            if mode == MODE_BLACKHOLE:
                sound.play('blackhole', 0.7)
                hud.flash("BLACK HOLE", (255, 200, 60), 2.0)

            elif mode == MODE_GALAXY:
                sound.play('galaxy', 0.55)
                hud.flash("GALAXY SWIRL", (160, 90, 255), 1.8)

            elif mode == MODE_NORMAL and prev_mode == MODE_BLACKHOLE:
                hud.flash("PRESS SPACE TO RESPAWN", (100, 220, 255), 3.0)

        # ── Supernova trigger ───────────────────────────────────────────
        if supernova_now:
            sound.play('supernova', 0.9)
            shockwaves.trigger(WIDTH//2, HEIGHT//2)
            flash_white = 1.0
            hud.flash("SUPERNOVA!", (255, 80, 70), 2.5)

        # ── Scan logic ──────────────────────────────────────────────────
        if scanning:
            scan_y += scan_speed
            if scan_y >= HEIGHT:
                scanning = False
                if _respawn_pending and len(targets) > 0:
                    pos       = spawn_on_skeleton(num_p, targets)
                    vel[:]    = 0
                    colors[:] = [0.0, 200.0, 255.0]
                    active[:] = True
                    _respawn_pending = False
                elif _respawn_pending:
                    pos       = np.random.rand(num_p, 2) * [WIDTH, HEIGHT]
                    vel[:]    = 0
                    active[:] = True
                    _respawn_pending = False
            else:
                active[pos[:, 1] < scan_y] = True

        # ── Update effects ──────────────────────────────────────────────
        accretion.update(
            mode == MODE_BLACKHOLE,
            bh_center if bh_center is not None else np.array([WIDTH/2.0, HEIGHT/2.0]),
            dt)
        shockwaves.update(dt)
        galaxy_ov.update(mode == MODE_GALAXY, dt)
        hud.update(dt, mode, fps, num_p, person, scanning, scan_y)

        # ── Physics ─────────────────────────────────────────────────────
        G       = 5000.0
        DAMPING = 0.95
        AURA    = 40.0
        phys_targets = targets.copy() if len(targets) > 0 else np.array([])

        if supernova_now or mode == MODE_SUPERNOVA:
            G             = -34000.0
            DAMPING       = 0.98
        elif mode == MODE_GALAXY:
            G       = 7500.0
            DAMPING = 0.96

        # ── BLACK HOLE PHYSICS (completely rewritten) ───────────────────
        if mode == MODE_BLACKHOLE and bh_center is not None:
            if np.any(active):
                ap  = pos[active]
                av  = vel[active]

                bh  = bh_center.astype(np.float64)
                d   = bh - ap                                    # (K,2) to hole
                dsq = d[:,0]**2 + d[:,1]**2 + 1.0               # (K,)
                dn  = np.sqrt(dsq)                               # (K,) distance

                # Radial pull (inverse-square)
                fm   = BH_G / dsq                                # (K,)
                norm = d / dn[:,np.newaxis]                      # (K,2) unit vec

                # Tangential swirl (perpendicular, same magnitude fraction)
                tang = np.stack([-d[:,1], d[:,0]], axis=1)       # (K,2) perp
                tn   = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
                tang = tang / tn

                acc  = norm * fm[:,np.newaxis] + tang * (fm * BH_SWIRL)[:,np.newaxis]
                av   = av * BH_DAMPING + acc * dt

                # Speed cap — prevents teleportation on next frame
                spd  = np.linalg.norm(av, axis=1, keepdims=True)
                av   = np.where(spd > BH_SPEED_CAP, av / spd * BH_SPEED_CAP, av)

                ap  += av * dt * TARGET_FPS

                pos[active] = ap
                vel[active] = av

            # Drain: consume particles within BH_CORE_RADIUS
            if np.any(active):
                aidx  = np.where(active)[0]
                dists = np.linalg.norm(pos[aidx] - bh_center, axis=1)
                consumed = aidx[dists < BH_CORE_RADIUS]
                if len(consumed):
                    active[consumed] = False
                    vel[consumed]    = 0

        else:
            # ── All other modes: normal N-body ──────────────────────────
            if len(phys_targets) > 0 and np.any(active):
                ap = pos[active]
                av = vel[active]

                if len(phys_targets) == 1:
                    diffs = phys_targets[0] - ap
                    dsq   = np.sum(diffs**2, axis=1) + 1.0
                    d_    = np.sqrt(dsq)
                    fm    = G / dsq
                    acc   = (diffs / d_[:,np.newaxis]) * fm[:,np.newaxis]
                else:
                    da  = phys_targets[np.newaxis,:,:] - ap[:,np.newaxis,:]
                    dsa = np.sum(da**2, axis=2)
                    ci  = np.argmin(dsa, axis=1)
                    cd  = da[np.arange(len(ap)), ci]
                    cds = dsa[np.arange(len(ap)), ci] + 20.0
                    cdd = np.sqrt(cds)
                    if supernova_now or mode == MODE_SUPERNOVA:
                        fm = G / cds
                    else:
                        err = cdd - AURA
                        fm  = np.clip(0.15*err, -10.0, 10.0)
                        fm += np.random.randn(len(ap)) * 0.5
                    acc = (cd / cdd[:,np.newaxis]) * fm[:,np.newaxis]

                av += acc
                av *= DAMPING
                ap += av
                pos[active] = ap
                vel[active] = av

        colors = colors * 0.88 + np.array(tgt_col, dtype=float) * 0.12

        # ══════════════════════════════════════════════
        #  RENDER
        # ══════════════════════════════════════════════

        # Layer 0: background photo
        screen.blit(bg, (0, 0))

        # Layer 1: darkening vignette
        dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dark.fill((0, 0, 12, 105))
        screen.blit(dark, (0, 0))

        # Layer 2: motion trail
        trail_surf.fill((0, 0, 0, 18))
        screen.blit(trail_surf, (0, 0))

        # Layer 3: galaxy spiral
        galaxy_ov.draw(screen)

        # Layer 4: scan line
        if scanning:
            sc = lerp_col((0,180,255),(0,255,100), abs(math.sin(now*5)))
            pygame.draw.line(screen, sc, (0,int(scan_y)),(WIDTH,int(scan_y)), 2)
            gs = pygame.Surface((WIDTH,16),pygame.SRCALPHA)
            gs.fill((*sc,30))
            screen.blit(gs,(0,int(scan_y)-8))

        # Layer 5: particles
        active_idx = np.where(active)[0]
        for i in active_idx:
            x, y = int(pos[i,0]), int(pos[i,1])
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                continue
            spd  = float(np.linalg.norm(vel[i]))
            size = 3 if spd > 10 else 2
            c    = (clamp(int(colors[i,0]),0,255),
                    clamp(int(colors[i,1]),0,255),
                    clamp(int(colors[i,2]),0,255))
            try:
                pygame.gfxdraw.aacircle(screen, x, y, size, c)
                pygame.gfxdraw.filled_circle(screen, x, y, size, c)
            except (ValueError, OverflowError):
                pass
            # Velocity streaks during supernova
            if spd > 14 and mode == MODE_SUPERNOVA:
                vn = vel[i]/(spd+1e-6)*min(spd*1.1, 18)
                pygame.draw.line(screen,(*c,90),(x,y),(int(x-vn[0]),int(y-vn[1])),1)

        # Layer 6: accretion disk
        accretion.draw(screen)

        # Layer 7: shockwaves
        shockwaves.draw(screen)

        # Layer 8: white flash (supernova)
        if flash_white > 0:
            fs = pygame.Surface((WIDTH, HEIGHT))
            fs.fill((255,255,255))
            fs.set_alpha(int(255*flash_white))
            screen.blit(fs,(0,0))
            flash_white = max(0.0, flash_white - dt*5.5)

        # Layer 9: debug skeleton
        if debug_mode and len(targets) > 0:
            for tx,ty in targets:
                pygame.draw.circle(screen,(255,0,0),(int(tx),int(ty)),4)

        # Layer 10: HUD
        hud.draw(screen)

        pygame.display.flip()

    # Cleanup
    cap.release()
    try: pose.close()
    except Exception: pass
    pygame.quit()
    print("  Session ended.")


if __name__ == "__main__":
    main()
