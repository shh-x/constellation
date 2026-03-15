"""
╔══════════════════════════════════════════════════════╗
║        COSMIC SIMULATION ENGINE  — v7.0              ║
║                                                      ║
║  GESTURE ZONES with HYSTERESIS (no flickering):      ║
║  ─────────────────────────────────────────────────   ║
║  Enter BH  : dist < 180px  (hands clearly together)  ║
║  Exit  BH  : dist > 260px  (hysteresis gap)          ║
║                                                      ║
║  Enter SN  : dist > 620px  (arms clearly wide)       ║
║  Exit  SN  : dist < 540px  (hysteresis gap)          ║
║                                                      ║
║  BH needs 3 consecutive frames to activate           ║
║  BH needs 4 consecutive frames to deactivate         ║
║  (prevents flicker from single noisy frames)         ║
╚══════════════════════════════════════════════════════╝
Controls:
  SPACE  — rescan / respawn particles on skeleton
  M      — debug skeleton overlay
  S      — screenshot
  ESC    — quit
"""

import sys, os, math, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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

# ===================================================
#  GESTURE THRESHOLDS  —  with HYSTERESIS
#
#  Your resting arm distance is ~380-500px.
#  Your BH attempts reached 180-250px.
#  Your SN attempts reached 550-1000px.
#
#  Hysteresis: ENTER threshold is tighter than EXIT.
#  This prevents the "flickering" where a single noisy
#  frame kicks you out of a mode you just entered.
#
#  BLACK HOLE:
#    Enter when dist < BH_ENTER (180px) — clearly together
#    Exit  when dist > BH_EXIT  (260px) — clearly separated
#
#  SUPERNOVA:
#    Enter when dist > SN_ENTER (620px) — clearly apart
#    Exit  when dist < SN_EXIT  (540px) — clearly not apart
# ===================================================
BH_ENTER = 180    # px — must be THIS close to enter BH
BH_EXIT  = 260    # px — must be THIS far  to leave BH
SN_ENTER = 620    # px — must be THIS far  to enter SN
SN_EXIT  = 540    # px — must be THIS close to leave SN

BH_CONFIRM_FRAMES  = 3   # frames below BH_ENTER before activating
BH_RELEASE_FRAMES  = 4   # frames above BH_EXIT  before deactivating

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
        t = self._t(2.0)
        d = (np.sin(2*np.pi*55*t)*0.5 + np.sin(2*np.pi*110*t)*0.25
             + np.sin(2*np.pi*82.5*t)*0.2)
        self._sounds['ambient'] = self._make(self._env(d, 0.4, 0.4) * 0.3)

        t = self._t(1.4)
        freq = 80 * np.exp(-t * 0.8)
        bh   = np.sin(2*np.pi * np.cumsum(freq) / self.SR)
        bh  += np.sin(2*np.pi*40*t)*0.4 + np.random.randn(len(t))*0.04
        self._sounds['blackhole'] = self._make(self._env(bh, 0.05, 0.4) * 0.65)

        t  = self._t(0.6)
        sn = (np.random.randn(len(t))*np.exp(-t*12)*0.6
              + np.sin(2*np.pi*400*t)*np.exp(-t*8)*0.3
              + (np.sin(2*np.pi*60*t)+np.sin(2*np.pi*90*t))*np.exp(-t*5)*0.5)
        self._sounds['supernova'] = self._make(self._env(sn, 0.001, 0.15) * 0.85)

        t    = self._t(0.35)
        scan = (np.sin(2*np.pi*1200*t)*np.exp(-t*6)
                + np.sin(2*np.pi*1800*t)*np.exp(-t*10)*0.4)
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
#  GESTURE RECOGNISER  v7  —  HYSTERESIS + DEBOUNCE
# ===================================================
class GestureRecogniser:
    SUPERNOVA_COOLDOWN = 3.0

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._slw = None
        self._srw = None
        self._K   = 0.55

        self._bh_enter_frames  = 0
        self._bh_exit_frames   = 0
        self._bh_active        = False

        self._sn_cooldown  = 0.0
        self._in_sn_zone   = False

        self._dbg = 0

    def _wrist_px(self, lm, idx):
        l   = lm[idx]
        vis = getattr(l, 'visibility', 1.0)
        pre = getattr(l, 'presence',   1.0)
        if vis < 0.05 or pre < 0.05:
            return None
        return np.array([l.x * self.w, l.y * self.h], dtype=np.float64)

    def update(self, landmarks, dt):
        self._sn_cooldown = max(0.0, self._sn_cooldown - dt)

        if landmarks is None:
            self._slw = self._srw = None
            self._bh_enter_frames = 0
            self._bh_exit_frames  = 0
            self._bh_active       = False
            self._in_sn_zone      = False
            return MODE_NORMAL, None, False

        lm = landmarks
        lw_raw = self._wrist_px(lm, 15)
        rw_raw = self._wrist_px(lm, 16)
        lw = lw_raw if lw_raw is not None else self._slw
        rw = rw_raw if rw_raw is not None else self._srw

        if lw is None or rw is None:
            self._bh_enter_frames = 0
            return MODE_NORMAL, None, False

        K = self._K
        if self._slw is None:
            self._slw = lw.copy()
            self._srw = rw.copy()
        else:
            self._slw = self._slw * (1-K) + lw * K
            self._srw = self._srw * (1-K) + rw * K

        dist_smooth = float(np.linalg.norm(self._slw - self._srw))
        dist_raw    = float(np.linalg.norm(lw - rw))

        # Entry uses min (catch any instant closeness)
        # Exit  uses max (require clearly separated)
        dist_for_enter = min(dist_raw, dist_smooth)
        dist_for_exit  = max(dist_raw, dist_smooth)
        dist_for_sn    = dist_smooth

        self._dbg += 1
        if self._dbg % 20 == 0:
            state = "BH" if self._bh_active else ("SN" if self._in_sn_zone else "--")
            print(f"  [GESTURE] enter={dist_for_enter:.0f}px  exit={dist_for_exit:.0f}px  "
                  f"sn={dist_for_sn:.0f}px  "
                  f"bh_in={self._bh_enter_frames}/{BH_CONFIRM_FRAMES}  "
                  f"bh_out={self._bh_exit_frames}/{BH_RELEASE_FRAMES}  "
                  f"[{state}]")

        supernova_fired = False

        # ── BLACK HOLE ──────────────────────────────────────────────────
        if not self._bh_active:
            if dist_for_enter < BH_ENTER:
                self._bh_enter_frames += 1
                self._bh_exit_frames   = 0
            else:
                self._bh_enter_frames  = max(0, self._bh_enter_frames - 1)

            if self._bh_enter_frames >= BH_CONFIRM_FRAMES:
                self._bh_active       = True
                self._bh_enter_frames = 0
                self._bh_exit_frames  = 0
                self._in_sn_zone      = False
                print(f"  [BH] ENTERED  dist={dist_for_enter:.0f}px")
        else:
            self._bh_enter_frames = 0
            if dist_for_exit > BH_EXIT:
                self._bh_exit_frames += 1
            else:
                self._bh_exit_frames  = max(0, self._bh_exit_frames - 1)

            if self._bh_exit_frames >= BH_RELEASE_FRAMES:
                self._bh_active      = False
                self._bh_exit_frames = 0
                print(f"  [BH] EXITED   dist={dist_for_exit:.0f}px")

        if self._bh_active:
            bh_pos = (lw + rw) / 2.0
            return MODE_BLACKHOLE, bh_pos, False

        # ── SUPERNOVA ───────────────────────────────────────────────────
        if dist_for_sn > SN_ENTER:
            if not self._in_sn_zone:
                self._in_sn_zone = True
                if self._sn_cooldown <= 0:
                    supernova_fired   = True
                    self._sn_cooldown = self.SUPERNOVA_COOLDOWN
                    print(f"  [SN] FIRED  dist={dist_for_sn:.0f}px")
            return (MODE_SUPERNOVA if supernova_fired else MODE_NORMAL), None, supernova_fired

        elif dist_for_sn < SN_EXIT:
            self._in_sn_zone = False

        return MODE_NORMAL, None, False

    @property
    def live_dist(self):
        if self._slw is None or self._srw is None:
            return 0.0
        return float(np.linalg.norm(self._slw - self._srw))


# ===================================================
#  PARTICLE SPAWNER
# ===================================================
def spawn_on_skeleton(num_p, targets):
    if targets is None or len(targets) == 0:
        return np.random.rand(num_p, 2) * [WIDTH, HEIGHT]
    idx      = np.random.randint(0, len(targets), num_p)
    pos      = targets[idx].copy() + np.random.randn(num_p, 2) * 20.0
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
        for r, th, col in [
            (140, 24, (255,  90,  10)),
            (108, 16, (255, 170,  30)),
            ( 76, 11, (255, 240, 110)),
            ( 48,  7, (255, 255, 210)),
        ]:
            pygame.draw.ellipse(s, (*col, int(clamp(190*a, 0, 255))),
                                (cx-r, cy-r//3, r*2, r*2//3), th)
        pygame.draw.circle(s, (  0,   0,   0, int(255*a)), (cx, cy), 32)
        pygame.draw.circle(s, (200, 200, 255, int(140*a)), (cx, cy), 40, 2)
        for gr, ga in [(22, 80), (16, 120), (10, 180)]:
            pygame.draw.circle(s, (255, 140, 0, int(ga*a)), (cx, cy), gr)
        surf.blit(s, (0, 0))


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
            a  = int(clamp(w['alpha'], 0, 255))
            r  = int(w['r'])
            if r <= 0: continue
            th = max(1, int(5*(1 - w['r']/w['maxr'])) + 1)
            pygame.draw.circle(s, (*w['col'], a),
                               (int(w['cx']), int(w['cy'])), r, th)
        surf.blit(s, (0, 0))


# ===================================================
#  HUD
# ===================================================
class HUD:
    COL_CYAN  = (100, 220, 255)
    COL_WHITE = (230, 235, 255)
    COL_GOLD  = (255, 200,  60)
    COL_RED   = (255,  80,  70)
    COL_PANEL = ( 10,  18,  40, 165)

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
        self.wrist_px       = 0.0

        pygame.font.init()
        self._f_flash = pygame.font.SysFont("Courier New", 52, bold=True)
        self._f_med   = pygame.font.SysFont("Courier New", 20, bold=True)
        self._f_small = pygame.font.SysFont("Courier New", 13)
        self._f_tiny  = pygame.font.SysFont("Courier New", 11)

    def update(self, dt, mode, fps, particles, person, scanning, scan_y, wrist_px=0):
        self._t             += dt
        self.mode            = mode
        self.fps             = fps
        self.particles       = particles
        self.person          = person
        self.scanning        = scanning
        self.scan_pct        = scan_y / HEIGHT
        self.wrist_px        = wrist_px
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

        CL = 32
        for (cx2, cy2), (dx, dy) in [
            ((8, 8),(1, 1)), ((self.w-8, 8),(-1, 1)),
            ((8, self.h-8),(1,-1)), ((self.w-8, self.h-8),(-1,-1))
        ]:
            ca = int(210*pulse)
            pygame.draw.line(ov, (*mc,ca), (cx2,cy2), (cx2+dx*CL, cy2), 2)
            pygame.draw.line(ov, (*mc,ca), (cx2,cy2), (cx2, cy2+dy*CL), 2)
            pygame.draw.circle(ov, (*mc,ca), (cx2,cy2), 2)

        # Status panel
        self._panel(ov, 8, 8, 310, 92)
        dot_col = (60,220,80) if self.person else (180,60,60)
        pygame.draw.circle(ov, (*dot_col,220), (22,22), 5)
        ring_r = int(8+3*abs(math.sin(self._t*4)))
        pygame.draw.circle(ov, (*dot_col,int(80*pulse)), (22,22), ring_r, 1)
        ov.blit(self._f_small.render(
            "ASTRAL ENTITY LOCKED" if self.person else "SCANNING VOID...",
            True, (*dot_col,210)), (32, 15))
        ov.blit(self._f_tiny.render(
            f"FPS {int(self.fps):>3}   STARS {self.particles:>4}",
            True, (*self.COL_WHITE,130)), (14, 36))

        # Live wrist distance bar with hysteresis zone markers
        if self.person:
            BAR_W   = 284
            BAR_X   = 14
            BAR_Y   = 54
            BAR_H   = 12
            MAX_VIS = SN_ENTER + 80

            filled = int(BAR_W * min(1.0, self.wrist_px / MAX_VIS))

            if self.wrist_px < BH_ENTER:
                bar_col = self.COL_GOLD
            elif self.wrist_px > SN_ENTER:
                bar_col = self.COL_RED
            elif self.wrist_px <= BH_EXIT:
                bar_col = (255, 160, 0)    # amber = BH hysteresis zone
            elif self.wrist_px >= SN_EXIT:
                bar_col = (255, 100, 60)   # orange = SN hysteresis zone
            else:
                bar_col = self.COL_CYAN

            pygame.draw.rect(ov, (25,30,60,200), (BAR_X, BAR_Y, BAR_W, BAR_H))
            if filled > 0:
                pygame.draw.rect(ov, (*bar_col,210), (BAR_X, BAR_Y, filled, BAR_H))

            # Threshold markers
            for px_val, col, solid in [
                (BH_ENTER, self.COL_GOLD, True),
                (BH_EXIT,  self.COL_GOLD, False),
                (SN_EXIT,  self.COL_RED,  False),
                (SN_ENTER, self.COL_RED,  True),
            ]:
                mx = BAR_X + int(BAR_W * px_val / MAX_VIS)
                a  = 220 if solid else 100
                pygame.draw.line(ov, (*col,a),
                                 (mx, BAR_Y-3), (mx, BAR_Y+BAR_H+2), 2 if solid else 1)

            ov.blit(self._f_tiny.render(
                f"wrists {int(self.wrist_px)}px   BH<{BH_ENTER}  SN>{SN_ENTER}",
                True, (*bar_col, 200)), (BAR_X, BAR_Y + BAR_H + 4))

        # Mode badge
        badge = self._f_med.render(self.mode, True, (*mc, int(235*pulse)))
        bw    = badge.get_width()
        bx    = self.w - bw - 16
        self._panel(ov, bx-8, 6, bw+16, 44)
        ov.blit(badge, (bx, 12))
        bar_fill = int(bw*abs(math.sin(self._t*1.6)))
        pygame.draw.rect(ov, (*mc,50),  (bx,38,bw,2))
        pygame.draw.rect(ov, (*mc,220), (bx,38,bar_fill,2))

        # Gesture guide
        gestures = [
            ("Hands Together", f"Black Hole  (<{BH_ENTER}px)"),
            ("Hands Apart",    f"Supernova   (>{SN_ENTER}px)"),
        ]
        gh = 14*len(gestures)+16
        gy = self.h - gh - 28
        self._panel(ov, 8, gy, 270, gh)
        for i, (trigger, effect) in enumerate(gestures):
            active_g = (("BLACK" in self.mode and "Black" in effect)
                        or ("SUPERNOVA" in self.mode and "Supernova" in effect))
            col_e = (*mc,210) if active_g else (120,140,170,170)
            ov.blit(self._f_tiny.render(f"{trigger}  ->  {effect}", True, col_e),
                    (16, gy+8+i*14))

        ctrl = "SPACE Rescan  |  M Debug  |  S Screenshot  |  ESC Quit"
        ov.blit(self._f_tiny.render(ctrl, True, (70,85,110,135)),
                (self.w//2 - self._f_tiny.size(ctrl)[0]//2, self.h-16))

        if self.scanning:
            bw2   = self.w - 40
            fill2 = int(bw2*self.scan_pct)
            pygame.draw.rect(ov, (*mc,35),  (20,self.h-46,bw2,3))
            pygame.draw.rect(ov, (*mc,200), (20,self.h-46,fill2,3))
            if fill2 > 0:
                pygame.draw.circle(ov,(*mc,180),(20+fill2,self.h-44),4)
            lbl = "ASTRAL ENTITY SCAN IN PROGRESS"
            ov.blit(self._f_small.render(lbl,True,(60,220,80,180)),
                    (self.w//2-self._f_small.size(lbl)[0]//2,self.h-62))

        if self.flash_timer > 0:
            alpha = int(255 * min(1.0, self.flash_timer / 0.5))
            fc    = self.flash_col
            for off in [(3,3),(-3,3),(3,-3),(-3,-3)]:
                g = self._f_flash.render(self.flash_msg,True,(*fc,clamp(alpha//5,0,255)))
                ov.blit(g,(self.w//2-g.get_width()//2+off[0],
                           self.h//2-g.get_height()//2-60+off[1]))
            fs = self._f_flash.render(self.flash_msg,True,(*fc,clamp(alpha,0,255)))
            ov.blit(fs,(self.w//2-fs.get_width()//2,self.h//2-fs.get_height()//2-60))
            tw = fs.get_width()
            tx = self.w//2 - tw//2
            ty = self.h//2 - 60 + fs.get_height() + 4
            lf = int(tw * min(1.0,(2.2-self.flash_timer)/0.4))
            pygame.draw.rect(ov,(*fc,clamp(alpha//2,0,255)),(tx,ty,tw,2))
            pygame.draw.rect(ov,(*fc,clamp(alpha,0,255)),(tx,ty,lf,2))

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
    pygame.display.set_caption("COSMIC SIMULATION ENGINE  v7.0")
    clock  = pygame.time.Clock()

    print("=" * 62)
    print("  COSMIC SIMULATION ENGINE  v7.0  —  Hysteresis Edition")
    print(f"  BH: enter <{BH_ENTER}px  exit >{BH_EXIT}px  "
          f"({BH_CONFIRM_FRAMES} frames in, {BH_RELEASE_FRAMES} frames out)")
    print(f"  SN: enter >{SN_ENTER}px  exit <{SN_EXIT}px")
    print("=" * 62)

    print("  [1/5] Loading background ...")
    try:
        bg = pygame.transform.scale(
            pygame.image.load(BG_IMAGE_PATH).convert(), (WIDTH, HEIGHT))
        print("        OK")
    except Exception as e:
        print(f"        WARNING: {e}  — solid fallback")
        bg = pygame.Surface((WIDTH, HEIGHT)); bg.fill((2, 6, 22))

    print("  [2/5] Opening webcam ...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("  ERROR: webcam unavailable."); sys.exit(1)

    print("  [3/5] Loading pose model ...")
    model_path            = os.path.join(_HERE, "pose_landmarker.task")
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.50,
        min_pose_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    pose = PoseLandmarker.create_from_options(options)

    print("  [4/5] Creating subsystems ...")
    gesture    = GestureRecogniser(WIDTH, HEIGHT)
    accretion  = AccretionDisk()
    shockwaves = ShockwaveManager()
    hud        = HUD(WIDTH, HEIGHT)

    print("  [5/5] Building audio ...")
    sound = SoundManager()

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
    _respawn_flashed = False

    trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    BH_G           = 180000.0
    BH_DAMPING     = 0.88
    BH_CORE_RADIUS = 28
    BH_SWIRL       = 0.18
    BH_SPEED_CAP   = 900.0

    print("=" * 62)
    print("  READY — Step into the void.")
    print(f"  Clasp hands (<{BH_ENTER}px) → BLACK HOLE")
    print(f"  Spread arms  (>{SN_ENTER}px) → SUPERNOVA")
    print("  Gold bar = BH zone  |  Red bar = SN zone")
    print("=" * 62)

    while running:
        now    = time.time()
        dt     = clamp(now - prev_t, 0.001, 0.05)
        prev_t = now
        clock.tick(TARGET_FPS)
        fps = clock.get_fps()

        if fps < 26 and num_p > 400:
            cut    = min(40, num_p-400)
            num_p -= cut
            pos    = pos[:num_p];    vel    = vel[:num_p]
            active = active[:num_p]; colors = colors[:num_p]

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
                    _respawn_flashed = False
                    sound.play('scan')
                elif event.key == pygame.K_m:
                    debug_mode = not debug_mode
                elif event.key == pygame.K_s:
                    fname = f"cosmic_{int(time.time())}.png"
                    pygame.image.save(screen, fname)
                    hud.screenshot_msg = 2.8
                    print(f"  Screenshot -> {fname}")

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

            lw_l = lms[15]; rw_l = lms[16]
            try:
                if getattr(lw_l,'visibility',0)>0.3 and getattr(rw_l,'visibility',0)>0.3:
                    n_y     = clamp((lw_l.y + rw_l.y)/2, 0, 1)
                    tgt_col = [int(lerp(0,200,n_y)),
                               int(lerp(200,80,n_y)),
                               int(lerp(255,255,n_y))]
            except Exception:
                pass

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
                        f = step / NUM_INTERP
                        interp.append(p1*(1-f) + p2*f)
            all_pts = base + interp
            targets = np.array(all_pts) if all_pts else np.array([])

        else:
            mode = MODE_NORMAL
            gesture.update(None, dt)

        # Mode-change events — fire ONCE per real transition
        if mode != last_mode:
            prev_mode = last_mode
            last_mode = mode

            if mode == MODE_BLACKHOLE:
                sound.play('blackhole', 0.7)
                hud.flash("BLACK HOLE", (255, 200, 60), 2.0)
                _respawn_flashed = False

            elif mode == MODE_NORMAL and prev_mode == MODE_BLACKHOLE:
                if not _respawn_flashed:
                    hud.flash("PRESS SPACE TO RESPAWN", (100, 220, 255), 3.0)
                    _respawn_flashed = True

        if supernova_now:
            sound.play('supernova', 0.9)
            shockwaves.trigger(WIDTH//2, HEIGHT//2)
            flash_white = 1.0
            hud.flash("SUPERNOVA!", (255, 80, 70), 2.5)

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

        accretion.update(
            mode == MODE_BLACKHOLE,
            bh_center if bh_center is not None else np.array([WIDTH/2.0, HEIGHT/2.0]),
            dt)
        shockwaves.update(dt)
        hud.update(dt, mode, fps, num_p, person, scanning, scan_y, gesture.live_dist)

        G       = 5000.0
        DAMPING = 0.95
        AURA    = 40.0
        phys_targets = targets.copy() if len(targets) > 0 else np.array([])

        if supernova_now or mode == MODE_SUPERNOVA:
            G       = -34000.0
            DAMPING = 0.98

        if mode == MODE_BLACKHOLE and bh_center is not None:
            if np.any(active):
                ap  = pos[active]; av = vel[active]
                bh  = bh_center.astype(np.float64)
                d   = bh - ap
                dsq = d[:,0]**2 + d[:,1]**2 + 1.0
                dn  = np.sqrt(dsq)
                fm   = BH_G / dsq
                norm = d / dn[:,np.newaxis]
                tang = np.stack([-d[:,1], d[:,0]], axis=1)
                tn   = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
                tang = tang / tn
                acc  = norm*fm[:,np.newaxis] + tang*(fm*BH_SWIRL)[:,np.newaxis]
                av   = av*BH_DAMPING + acc*dt
                spd  = np.linalg.norm(av, axis=1, keepdims=True)
                av   = np.where(spd > BH_SPEED_CAP, av/spd*BH_SPEED_CAP, av)
                ap  += av * dt * TARGET_FPS
                pos[active] = ap; vel[active] = av
            if np.any(active):
                aidx  = np.where(active)[0]
                dists = np.linalg.norm(pos[aidx] - bh_center, axis=1)
                consumed = aidx[dists < BH_CORE_RADIUS]
                if len(consumed):
                    active[consumed] = False; vel[consumed] = 0
        else:
            if len(phys_targets) > 0 and np.any(active):
                ap = pos[active]; av = vel[active]
                if len(phys_targets) == 1:
                    diffs = phys_targets[0] - ap
                    dsq   = np.sum(diffs**2,axis=1) + 1.0
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
                av += acc; av *= DAMPING; ap += av
                pos[active] = ap; vel[active] = av

        colors = colors * 0.88 + np.array(tgt_col, dtype=float) * 0.12

        screen.blit(bg, (0, 0))
        dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dark.fill((0, 0, 12, 105))
        screen.blit(dark, (0, 0))
        trail_surf.fill((0, 0, 0, 18))
        screen.blit(trail_surf, (0, 0))

        if scanning:
            sc = lerp_col((0,180,255),(0,255,100), abs(math.sin(now*5)))
            pygame.draw.line(screen, sc, (0,int(scan_y)),(WIDTH,int(scan_y)), 2)
            gs = pygame.Surface((WIDTH,16),pygame.SRCALPHA)
            gs.fill((*sc,30))
            screen.blit(gs,(0,int(scan_y)-8))

        active_idx = np.where(active)[0]
        for i in active_idx:
            x, y = int(pos[i,0]), int(pos[i,1])
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT): continue
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
            if spd > 14 and mode == MODE_SUPERNOVA:
                vn = vel[i]/(spd+1e-6)*min(spd*1.1,18)
                pygame.draw.line(screen,(*c,90),(x,y),(int(x-vn[0]),int(y-vn[1])),1)

        accretion.draw(screen)
        shockwaves.draw(screen)

        if flash_white > 0:
            fs = pygame.Surface((WIDTH, HEIGHT))
            fs.fill((255,255,255))
            fs.set_alpha(int(255*flash_white))
            screen.blit(fs,(0,0))
            flash_white = max(0.0, flash_white - dt*5.5)

        if debug_mode and len(targets) > 0:
            for tx,ty in targets:
                pygame.draw.circle(screen,(255,0,0),(int(tx),int(ty)),4)

        hud.draw(screen)
        pygame.display.flip()

    cap.release()
    try: pose.close()
    except Exception: pass
    pygame.quit()
    print("  Session ended.")


if __name__ == "__main__":
    main()